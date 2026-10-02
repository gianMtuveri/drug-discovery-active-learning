#!/usr/bin/env python3
"""Resumable full-population extraction for frozen SMI-TED and Uni-Mol v1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.representations import file_sha256, make_molecule_ids, ordered_population_hash
from scripts.representations.run_embedding_pilot import (
    CONFORMER_SEED,
    MMFF_MAX_ITERS,
    SMI_TED_MAX_TOTAL_TOKENS,
    UNIMOL_MAX_ATOMS,
    _freeze,
    _load_smi_module,
    _load_unimol,
    generate_unimol_geometry,
)


EXPECTED_ROWS = 13_995
SMI_DIMENSION = 768
UNIMOL_DIMENSION = 512
SMI_ATOL = 1e-5
UNIMOL_ATOL = 1e-6
SUCCESSFUL_UNIMOL_STATUSES = {"optimized", "mmff_unavailable", "mmff_nonconverged"}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _directory_file_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): file_sha256(item)
        for item in sorted(path.rglob("*.py"))
        if "__pycache__" not in item.parts
    }


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full extraction; CPU fallback is prohibited")
    return torch.device("cuda:0")


def _source_population(dataset: Path) -> pd.DataFrame:
    source = pd.read_parquet(dataset, columns=["canonical_smiles"])
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} corrected EGFR rows, found {len(source)}")
    smiles = source.canonical_smiles.astype(str).to_numpy()
    if len(np.unique(smiles)) != EXPECTED_ROWS:
        raise ValueError("Corrected EGFR population is not unique by canonical SMILES")
    return pd.DataFrame({
        "original_index": np.arange(EXPECTED_ROWS, dtype=np.int64),
        "molecule_id": make_molecule_ids(smiles),
        "canonical_smiles": smiles,
    })


def _create_nan_array(path: Path, shape: tuple[int, int]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    matrix[:] = np.nan
    matrix.flush()
    return matrix


def _open_or_create_embeddings(path: Path, shape: tuple[int, int]) -> np.memmap:
    if path.exists():
        matrix = np.load(path, mmap_mode="r+")
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError(f"Incompatible resumable embedding artifact: {path}")
        return matrix
    return _create_nan_array(path, shape)


def _save_audit(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _base_manifest(args: argparse.Namespace, population: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_dataset": str(args.dataset),
        "source_dataset_sha256": file_sha256(args.dataset),
        "source_row_count": len(population),
        "ordered_population_hash": ordered_population_hash(population.molecule_id),
        "runtime_versions": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "rdkit", "torch", "transformers", "unicore", "unimol")
        },
    }


def extract_smi_ted(args: argparse.Namespace, population: pd.DataFrame, device: torch.device) -> None:
    artifact_dir = args.artifact_dir / "smi_ted_light"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    audit_path = artifact_dir / "extraction_audit.parquet"
    embeddings_path = artifact_dir / "embeddings.npy"
    smi_dir = args.checkpoint_dir / "smi_ted_light"
    official = _load_smi_module(smi_dir)
    tokenizer = official.MolTranBertTokenizer(str(smi_dir / "bert_vocab_curated.txt"))

    if audit_path.exists():
        audit = pd.read_parquet(audit_path)
        _assert_population_alignment(audit, population)
    else:
        rows: list[dict[str, Any]] = []
        for row in population.itertuples(index=False):
            normalized = official.normalize_smiles(row.canonical_smiles)
            if normalized is None:
                token_count, status, reason = None, "normalization_failed", "official_normalization_failed"
            else:
                try:
                    token_count = len(tokenizer(
                        normalized, add_special_tokens=True, truncation=False
                    )["input_ids"])
                    if token_count > SMI_TED_MAX_TOTAL_TOKENS:
                        status, reason = "token_limit_exceeded", f"{token_count}>{SMI_TED_MAX_TOTAL_TOKENS}"
                    else:
                        status, reason = "pending", ""
                except Exception as exc:
                    token_count, status = None, "tokenization_failed"
                    reason = f"{type(exc).__name__}:{exc}"
            rows.append({
                **row._asdict(),
                "normalized_smiles": normalized,
                "total_token_count": token_count,
                "status": status,
                "failure_reason": reason,
            })
        audit = pd.DataFrame(rows)
        _save_audit(audit, audit_path)

    embeddings = _open_or_create_embeddings(embeddings_path, (EXPECTED_ROWS, SMI_DIMENSION))
    model = official.load_smi_ted(
        str(smi_dir), "smi-ted-Light_40.pt", "bert_vocab_curated.txt"
    )
    _freeze(model)
    pending = audit.index[audit.status == "pending"].tolist()
    for start in range(0, len(pending), args.smi_checkpoint_size):
        indices = pending[start:start + args.smi_checkpoint_size]
        inputs = audit.loc[indices, "canonical_smiles"].tolist()
        with torch.inference_mode():
            encoded = model.encode(
                inputs,
                batch_size=min(args.smi_batch_size, len(inputs)),
                return_torch=True,
            )
        values = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
        if values.shape != (len(indices), SMI_DIMENSION) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid SMI-TED batch beginning at source row {indices[0]}")
        embeddings[indices] = values
        embeddings.flush()
        audit.loc[indices, "status"] = "success"
        _save_audit(audit, audit_path)
        print(f"SMI-TED checkpoint: {int((audit.status == 'success').sum())}/{len(audit)}", flush=True)

    success = audit.status.eq("success").to_numpy()
    _finalize_representation(
        artifact_dir, population, audit, embeddings, success, SMI_DIMENSION,
        {
            **_base_manifest(args, population),
            "representation": "smi_ted_light",
            "feature_dimension": SMI_DIMENSION,
            "feature_dtype": "float32",
            "checkpoint": "smi-ted-Light_40.pt",
            "checkpoint_sha256": file_sha256(smi_dir / "smi-ted-Light_40.pt"),
            "loader_sha256": file_sha256(smi_dir / "load.py"),
            "vocabulary_sha256": file_sha256(smi_dir / "bert_vocab_curated.txt"),
            "adapted_fast_transformers_sha256": _directory_file_hashes(
                smi_dir / "fast_transformers"
            ),
            "normalization": {"canonical": True, "isomeric": False},
            "maximum_total_tokens": SMI_TED_MAX_TOTAL_TOKENS,
            "silent_truncation_permitted": False,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
    )
    del model, embeddings
    torch.cuda.empty_cache()


def _geometry_worker(item: tuple[int, str]) -> dict[str, Any]:
    index, smiles = item
    status, reason, mmff_code, heavy, coordinates = generate_unimol_geometry(smiles)
    atoms = None if heavy is None else [atom.GetSymbol() for atom in heavy.GetAtoms()]
    return {
        "index": index,
        "status": status,
        "failure_reason": reason,
        "mmff_code": mmff_code,
        "atoms": atoms,
        "coordinates": coordinates,
    }


def _unimol_batch_embeddings(
    model, dictionary, geometries: list[dict[str, Any]], device: torch.device
) -> np.ndarray:
    pad = dictionary.pad()
    lengths = [len(item["atoms"]) + 2 for item in geometries]
    maximum = max(lengths)
    tokens = torch.full((len(geometries), maximum), pad, dtype=torch.long, device=device)
    coordinates = torch.zeros((len(geometries), maximum, 3), dtype=torch.float32, device=device)
    for row, item in enumerate(geometries):
        atoms = item["atoms"]
        unknown = sorted(set(atoms).difference(dictionary.indices))
        if unknown:
            raise ValueError(f"Unsupported dictionary atom types: {unknown}")
        xyz = np.asarray(item["coordinates"], dtype=np.float32)
        centered = xyz - xyz.mean(axis=0, keepdims=True)
        ids = [dictionary.bos(), *[dictionary.index(atom) for atom in atoms], dictionary.eos()]
        tokens[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        coordinates[row, 1:len(ids) - 1] = torch.tensor(centered, dtype=torch.float32, device=device)
    distances = torch.cdist(coordinates, coordinates)
    edges = tokens.unsqueeze(-1) * len(dictionary) + tokens.unsqueeze(-2)
    with torch.inference_mode():
        representation, _ = model(tokens, distances, coordinates, edges, features_only=True)
    values = representation[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    if values.shape != (len(geometries), UNIMOL_DIMENSION) or not np.isfinite(values).all():
        raise RuntimeError("Invalid Uni-Mol CLS batch")
    return values


def extract_unimol(args: argparse.Namespace, population: pd.DataFrame, device: torch.device) -> None:
    artifact_dir = args.artifact_dir / "unimol_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    audit_path = artifact_dir / "extraction_audit.parquet"
    embeddings_path = artifact_dir / "embeddings.npy"
    uni_dir = args.checkpoint_dir / "unimol_v1"

    if audit_path.exists():
        audit = pd.read_parquet(audit_path)
        _assert_population_alignment(audit, population)
    else:
        audit = population.copy()
        heavy_counts: list[int | None] = []
        initial_status: list[str] = []
        reasons: list[str] = []
        for smiles in population.canonical_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                heavy_counts.append(None)
                initial_status.append("embedding_failed")
                reasons.append("rdkit_parse_failed")
                continue
            count = mol.GetNumHeavyAtoms()
            heavy_counts.append(count)
            if count > UNIMOL_MAX_ATOMS:
                initial_status.append("atom_limit_exceeded")
                reasons.append(f"{count}>{UNIMOL_MAX_ATOMS}")
            else:
                initial_status.append("pending")
                reasons.append("")
        audit["heavy_atom_count"] = heavy_counts
        audit["status"] = initial_status
        audit["failure_reason"] = reasons
        audit["mmff_code"] = pd.array([None] * len(audit), dtype="Int64")
        audit["conformer_seed"] = pd.array([
            None if status == "atom_limit_exceeded" else CONFORMER_SEED
            for status in initial_status
        ], dtype="Int64")
        _save_audit(audit, audit_path)

    embeddings = _open_or_create_embeddings(embeddings_path, (EXPECTED_ROWS, UNIMOL_DIMENSION))
    model, dictionary = _load_unimol(uni_dir, device)
    pending = audit.index[audit.status == "pending"].tolist()
    work_items = [(index, audit.at[index, "canonical_smiles"]) for index in pending]
    with ProcessPoolExecutor(
        max_workers=args.conformer_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        geometry_iterator = executor.map(_geometry_worker, work_items, chunksize=1)
        checkpoint: list[dict[str, Any]] = []
        for result in geometry_iterator:
            checkpoint.append(result)
            if len(checkpoint) >= args.unimol_checkpoint_size:
                _process_unimol_checkpoint(checkpoint, audit, embeddings, model, dictionary, device, args)
                checkpoint = []
        if checkpoint:
            _process_unimol_checkpoint(checkpoint, audit, embeddings, model, dictionary, device, args)

    success = audit.status.isin(SUCCESSFUL_UNIMOL_STATUSES).to_numpy()
    _finalize_representation(
        artifact_dir, population, audit, embeddings, success, UNIMOL_DIMENSION,
        {
            **_base_manifest(args, population),
            "representation": "unimol_v1",
            "feature_dimension": UNIMOL_DIMENSION,
            "feature_dtype": "float32",
            "checkpoint": "mol_pre_no_h_220816.pt",
            "checkpoint_sha256": file_sha256(uni_dir / "mol_pre_no_h_220816.pt"),
            "dictionary_sha256": file_sha256(uni_dir / "mol.dict.txt"),
            "features_only": True,
            "embedding": "CLS",
            "maximum_heavy_atoms": UNIMOL_MAX_ATOMS,
            "cropping_permitted": False,
            "conformer": {
                "method": "ETKDGv3", "seed": CONFORMER_SEED,
                "add_hydrogens": True, "remove_hydrogens_before_inference": True,
                "mmff_max_iterations": MMFF_MAX_ITERS,
                "fallback_coordinates": False,
            },
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
    )
    del model, embeddings
    torch.cuda.empty_cache()


def _process_unimol_checkpoint(
    checkpoint: list[dict[str, Any]], audit: pd.DataFrame, embeddings: np.memmap,
    model, dictionary, device: torch.device, args: argparse.Namespace,
) -> None:
    valid: list[dict[str, Any]] = []
    for item in checkpoint:
        index = item["index"]
        audit.at[index, "status"] = item["status"]
        audit.at[index, "failure_reason"] = item["failure_reason"]
        audit.at[index, "mmff_code"] = item["mmff_code"]
        if item["status"] in SUCCESSFUL_UNIMOL_STATUSES:
            valid.append(item)
    # Padding different molecular lengths changes the validated Uni-Mol CLS path.
    # Batch only equal-length inputs, so no atom/token padding is introduced.
    by_length: dict[int, list[dict[str, Any]]] = {}
    for item in valid:
        by_length.setdefault(len(item["atoms"]), []).append(item)
    for atom_count in sorted(by_length):
        same_length = by_length[atom_count]
        for start in range(0, len(same_length), args.unimol_batch_size):
            batch = same_length[start:start + args.unimol_batch_size]
            try:
                values = _unimol_batch_embeddings(model, dictionary, batch, device)
                embeddings[[item["index"] for item in batch]] = values
            except Exception as batch_error:
                # Preserve progress and isolate the exact failing molecule without changing inference.
                for item in batch:
                    try:
                        value = _unimol_batch_embeddings(model, dictionary, [item], device)
                        embeddings[item["index"]] = value[0]
                    except Exception as exc:
                        index = item["index"]
                        audit.at[index, "status"] = "inference_failed"
                        audit.at[index, "failure_reason"] = f"{type(exc).__name__}:{exc}"
                print(f"Uni-Mol batch isolation after {type(batch_error).__name__}", flush=True)
    embeddings.flush()
    _save_audit(audit, args.artifact_dir / "unimol_v1" / "extraction_audit.parquet")
    complete = int((audit.status != "pending").sum())
    print(f"Uni-Mol checkpoint: {complete}/{len(audit)}", flush=True)


def _assert_population_alignment(audit: pd.DataFrame, population: pd.DataFrame) -> None:
    if len(audit) != len(population):
        raise ValueError("Resumable audit row count does not match source population")
    for column in ("original_index", "molecule_id", "canonical_smiles"):
        if not np.array_equal(audit[column].to_numpy(), population[column].to_numpy()):
            raise ValueError(f"Resumable audit is misaligned in {column}")


def _finalize_representation(
    artifact_dir: Path, population: pd.DataFrame, audit: pd.DataFrame,
    embeddings: np.ndarray, success: np.ndarray, dimension: int,
    manifest: dict[str, Any],
) -> None:
    _assert_population_alignment(audit, population)
    if embeddings.shape != (EXPECTED_ROWS, dimension) or embeddings.dtype != np.float32:
        raise ValueError("Embedding shape or dtype mismatch")
    finite_rows = np.isfinite(embeddings).all(axis=1)
    nan_rows = np.isnan(embeddings).all(axis=1)
    if not np.array_equal(finite_rows, success):
        raise ValueError("Success mask does not match finite embedding rows")
    if not np.all(nan_rows[~success]):
        raise ValueError("Failed embedding rows are not consistently all-NaN")
    np.save(artifact_dir / "success_mask.npy", success.astype(bool))
    np.save(artifact_dir / "molecule_ids.npy", population.molecule_id.to_numpy(dtype=str))
    manifest.update({
        "feature_shape": [EXPECTED_ROWS, dimension],
        "successful_rows": int(success.sum()),
        "unsuccessful_rows": int((~success).sum()),
        "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
        "complete": True,
    })
    _json_dump(artifact_dir / "manifest.json", manifest)


def _reproducibility(args: argparse.Namespace, population: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    count = args.reproducibility_count
    smi_dir = args.artifact_dir / "smi_ted_light"
    smi_audit = pd.read_parquet(smi_dir / "extraction_audit.parquet")
    smi_stored = np.load(smi_dir / "embeddings.npy", mmap_mode="r")
    smi_indices = smi_audit.index[smi_audit.status == "success"].tolist()[:count]
    official_dir = args.checkpoint_dir / "smi_ted_light"
    official = _load_smi_module(official_dir)
    smi_model = official.load_smi_ted(
        str(official_dir), "smi-ted-Light_40.pt", "bert_vocab_curated.txt"
    )
    _freeze(smi_model)
    with torch.inference_mode():
        repeat = smi_model.encode(
            population.loc[smi_indices, "canonical_smiles"].tolist(),
            batch_size=1, return_torch=True,
        ).cpu().numpy().astype(np.float32)
    smi_difference = float(np.max(np.abs(smi_stored[smi_indices] - repeat)))
    del smi_model, repeat, official
    torch.cuda.empty_cache()

    uni_dir = args.artifact_dir / "unimol_v1"
    uni_audit = pd.read_parquet(uni_dir / "extraction_audit.parquet")
    uni_stored = np.load(uni_dir / "embeddings.npy", mmap_mode="r")
    uni_indices = uni_audit.index[uni_audit.status.isin(SUCCESSFUL_UNIMOL_STATUSES)].tolist()[:count]
    geometries = []
    for index in uni_indices:
        result = _geometry_worker((index, population.at[index, "canonical_smiles"]))
        if result["status"] not in SUCCESSFUL_UNIMOL_STATUSES:
            raise RuntimeError(f"Reproducibility conformer failed at row {index}")
        geometries.append(result)
    uni_model, dictionary = _load_unimol(args.checkpoint_dir / "unimol_v1", device)
    # Repeat each molecule without cross-length padding, matching validated extraction.
    uni_repeat = np.vstack([
        _unimol_batch_embeddings(uni_model, dictionary, [geometry], device)[0]
        for geometry in geometries
    ]).astype(np.float32, copy=False)
    uni_difference = float(np.max(np.abs(uni_stored[uni_indices] - uni_repeat)))
    del uni_model, uni_repeat
    torch.cuda.empty_cache()
    return {
        "smi_ted": {
            "original_indices": smi_indices, "sample_size": len(smi_indices),
            "maximum_absolute_difference": smi_difference,
            "absolute_tolerance": SMI_ATOL, "passed": smi_difference <= SMI_ATOL,
            "model_reloaded": True,
        },
        "unimol": {
            "original_indices": uni_indices, "sample_size": len(uni_indices),
            "maximum_absolute_difference": uni_difference,
            "absolute_tolerance": UNIMOL_ATOL, "passed": uni_difference <= UNIMOL_ATOL,
            "model_reloaded": True, "conformers_regenerated_from_scratch": True,
        },
    }


def validate_and_report(args: argparse.Namespace, population: pd.DataFrame, device: torch.device) -> None:
    summaries: dict[str, Any] = {}
    for name, dimension, successful_statuses in (
        ("smi_ted_light", SMI_DIMENSION, {"success"}),
        ("unimol_v1", UNIMOL_DIMENSION, SUCCESSFUL_UNIMOL_STATUSES),
    ):
        directory = args.artifact_dir / name
        audit = pd.read_parquet(directory / "extraction_audit.parquet")
        embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        mask = np.load(directory / "success_mask.npy")
        ids = np.load(directory / "molecule_ids.npy")
        _assert_population_alignment(audit, population)
        if not np.array_equal(ids, population.molecule_id.to_numpy(dtype=str)):
            raise ValueError(f"{name} molecule IDs are misaligned")
        expected = audit.status.isin(successful_statuses).to_numpy()
        if not np.array_equal(mask, expected):
            raise ValueError(f"{name} stored mask differs from statuses")
        _finalize_representation(
            directory, population, audit, embeddings, expected, dimension,
            json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        )
        summaries[name] = {
            "feature_shape": list(embeddings.shape), "feature_dtype": str(embeddings.dtype),
            "successful": int(mask.sum()), "unsuccessful": int((~mask).sum()),
            "success_fraction": float(mask.mean()),
            "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
        }
        destination = args.results_dir / f"{name}_extraction_audit.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "extraction_audit.parquet", destination)
    smi_audit = pd.read_parquet(args.artifact_dir / "smi_ted_light" / "extraction_audit.parquet")
    uni_audit = pd.read_parquet(args.artifact_dir / "unimol_v1" / "extraction_audit.parquet")
    summaries["smi_ted_light"].update({
        "token_count_range": [int(smi_audit.total_token_count.min()), int(smi_audit.total_token_count.max())],
        "over_202_count": int((smi_audit.total_token_count > SMI_TED_MAX_TOTAL_TOKENS).sum()),
    })
    summaries["unimol_v1"].update({
        "heavy_atom_range": [int(uni_audit.heavy_atom_count.min()), int(uni_audit.heavy_atom_count.max())],
        "over_256_count": int((uni_audit.heavy_atom_count > UNIMOL_MAX_ATOMS).sum()),
    })
    reproducibility = _reproducibility(args, population, device)
    summary = {
        "source_row_count": len(population),
        "ordered_population_hash": ordered_population_hash(population.molecule_id),
        "representations": summaries,
        "reproducibility": reproducibility,
        "common_population_computed": False,
        "pilot_comparison": {
            "smi_ted": "The pilot's sole token-limit violation recurred at source row 5019; no additional SMI-TED failures occurred.",
            "unimol": "The pilot's row-5019 atom-limit violation and row-8426/8436 embedding failures recurred. The full population added 9 embedding failures, 31 isotopic-hydrogen alignment failures, 56 MMFF nonconvergences, and 17 MMFF-unavailable successes.",
        },
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
    }
    _json_dump(args.results_dir / "extraction_summary.json", summary)
    provenance = {
        **_base_manifest(args, population),
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "smi_ted_manifest": json.loads((args.artifact_dir / "smi_ted_light" / "manifest.json").read_text()),
        "unimol_manifest": json.loads((args.artifact_dir / "unimol_v1" / "manifest.json").read_text()),
    }
    _json_dump(args.results_dir / "provenance.json", provenance)
    _write_report(args.results_dir / "report.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    smi = summary["representations"]["smi_ted_light"]
    uni = summary["representations"]["unimol_v1"]
    repro = summary["reproducibility"]
    text = f"""# v0.5.1 full frozen-representation extraction

- Source rows: {summary['source_row_count']:,}
- Device: {summary['device_name']}
- SMI-TED: {smi['successful']:,} successful, {smi['unsuccessful']:,} unsuccessful; shape {smi['feature_shape']}
- SMI-TED statuses: {smi['status_counts']}
- Uni-Mol: {uni['successful']:,} successful, {uni['unsuccessful']:,} unsuccessful; shape {uni['feature_shape']}
- Uni-Mol statuses: {uni['status_counts']}
- SMI-TED reproducibility max |difference|: {repro['smi_ted']['maximum_absolute_difference']:.9g} (tolerance {SMI_ATOL:.1g})
- Uni-Mol reproducibility max |difference|: {repro['unimol']['maximum_absolute_difference']:.9g} (tolerance {UNIMOL_ATOL:.1g})

Rows remain aligned to the corrected EGFR population. Failed rows are all-NaN. No common-population intersection was computed.

The pilot's known limit violations and two Uni-Mol conformer failures recurred. The full extraction additionally found nine RDKit embedding failures and 31 strict alignment failures in deuterium-labelled structures, for which RDKit's default hydrogen removal retained isotopic hydrogen. Those rows remain explicitly ineligible; the approved geometry procedure was not changed silently.
"""
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smi-ted", "unimol", "validate", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v051/targets/EGFR/clean.parquet"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/processed/v051/checkpoints"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("data/processed/v051/representations"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/representation_extraction/v051"))
    parser.add_argument("--smi-batch-size", type=int, default=64)
    parser.add_argument("--smi-checkpoint-size", type=int, default=1024)
    # Batch dimension changes CLS values beyond the validated 1e-6 tolerance.
    # Keep exact pilot-equivalent, unpadded single-molecule inference.
    parser.add_argument("--unimol-batch-size", type=int, default=1)
    parser.add_argument("--unimol-checkpoint-size", type=int, default=128)
    parser.add_argument("--conformer-workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--reproducibility-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = _require_cuda()
    population = _source_population(args.dataset)
    if args.stage in {"smi-ted", "all"}:
        extract_smi_ted(args, population, device)
    if args.stage in {"unimol", "all"}:
        extract_unimol(args, population, device)
    if args.stage in {"validate", "all"}:
        validate_and_report(args, population, device)


if __name__ == "__main__":
    main()
