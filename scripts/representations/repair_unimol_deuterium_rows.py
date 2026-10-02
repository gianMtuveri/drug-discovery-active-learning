#!/usr/bin/env python3
"""Surgically repair the 31 audited Uni-Mol deuterium rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.representations.extract_full_frozen_representations import (
    SUCCESSFUL_UNIMOL_STATUSES,
    UNIMOL_ATOL,
    _unimol_batch_embeddings,
)
from scripts.representations.run_embedding_pilot import (
    CONFORMER_SEED,
    _load_unimol,
    generate_unimol_geometry,
)
from src.data.representations import file_sha256, make_molecule_ids


TARGET_COUNT = 31
PRODUCTION_FILES = (
    "embeddings.npy", "success_mask.npy", "molecule_ids.npy",
    "extraction_audit.parquet", "manifest.json",
)


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hashes(directory: Path) -> dict[str, str]:
    return {name: file_sha256(directory / name) for name in PRODUCTION_FILES}


def _backup_production(directory: Path, backup: Path) -> dict[str, str]:
    if backup.exists():
        raise FileExistsError(f"Refusing to reuse existing repair backup: {backup}")
    backup.mkdir(parents=True)
    for name in PRODUCTION_FILES:
        shutil.copy2(directory / name, backup / name)
    source_hashes = _hashes(directory)
    backup_hashes = _hashes(backup)
    if source_hashes != backup_hashes:
        raise RuntimeError("Pre-repair backup hashes do not match production")
    _json_dump(backup / "baseline_hashes.json", source_hashes)
    return source_hashes


def _write_audit(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _assert_unchanged_rows(
    production_dir: Path, backup_dir: Path, target_indices: np.ndarray,
) -> None:
    total = len(np.load(backup_dir / "success_mask.npy"))
    non_target = np.ones(total, dtype=bool)
    non_target[target_indices] = False

    before_embeddings = np.load(backup_dir / "embeddings.npy", mmap_mode="r")
    after_embeddings = np.load(production_dir / "embeddings.npy", mmap_mode="r")
    if not np.array_equal(
        before_embeddings[non_target], after_embeddings[non_target], equal_nan=True
    ):
        raise RuntimeError("A non-target embedding row changed")

    before_mask = np.load(backup_dir / "success_mask.npy")
    after_mask = np.load(production_dir / "success_mask.npy")
    if not np.array_equal(before_mask[non_target], after_mask[non_target]):
        raise RuntimeError("A non-target success-mask entry changed")

    before_ids = np.load(backup_dir / "molecule_ids.npy")
    after_ids = np.load(production_dir / "molecule_ids.npy")
    if not np.array_equal(before_ids, after_ids):
        raise RuntimeError("Molecule IDs changed")

    before_audit = pd.read_parquet(backup_dir / "extraction_audit.parquet")
    after_audit = pd.read_parquet(production_dir / "extraction_audit.parquet")
    for column in ("original_index", "molecule_id", "canonical_smiles"):
        if not np.array_equal(before_audit[column].to_numpy(), after_audit[column].to_numpy()):
            raise RuntimeError(f"Audit alignment changed in {column}")
    pd.testing.assert_frame_equal(
        before_audit.loc[non_target].reset_index(drop=True),
        after_audit.loc[non_target].reset_index(drop=True),
        check_exact=True,
    )


def _update_reports(
    args: argparse.Namespace, audit: pd.DataFrame, before_hashes: dict[str, str],
    target_indices: list[int], reproducibility: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = args.production_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    success = audit.status.isin(SUCCESSFUL_UNIMOL_STATUSES)
    manifest.update({
        "successful_rows": int(success.sum()),
        "unsuccessful_rows": int((~success).sum()),
        "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
        "hydrogen_removal": "all_atomic_number_1_atoms_including_isotopes",
        "deuterium_representation_limitation": (
            "The no-H representation removes hydrogen isotopes; isotope mass is not represented."
        ),
        "deuterium_repair": {
            "repaired_row_count": len(target_indices),
            "original_indices": target_indices,
            "conformer_seed": CONFORMER_SEED,
            "inference": "single_molecule_features_only_cls",
        },
    })
    _json_dump(manifest_path, manifest)

    extraction_summary_path = args.results_dir / "extraction_summary.json"
    summary = json.loads(extraction_summary_path.read_text(encoding="utf-8"))
    uni = summary["representations"]["unimol_v1"]
    uni.update({
        "successful": int(success.sum()),
        "unsuccessful": int((~success).sum()),
        "success_fraction": float(success.mean()),
        "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
    })
    summary["reproducibility"]["unimol_deuterium_repair"] = reproducibility
    summary["pilot_comparison"]["unimol"] = (
        "The original atom-limit and deterministic RDKit embedding failures remain. "
        "The 31 audited deuterium rows were repaired by enforcing the pinned no-H semantics."
    )
    summary["unimol_no_h_limitation"] = (
        "All hydrogen isotopes are removed before inference; deuterium identity is not represented. "
        "Original canonical structures are unchanged."
    )
    _json_dump(extraction_summary_path, summary)

    provenance_path = args.results_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["unimol_manifest"] = manifest
    provenance["deuterium_repair"] = {
        "baseline_hashes": before_hashes,
        "repaired_original_indices": target_indices,
        "hydrogen_removal": "atomic_number_1",
        "representation_limitation": "deuterium isotope mass is not represented",
        "reproducibility": reproducibility,
    }
    _json_dump(provenance_path, provenance)

    smi = summary["representations"]["smi_ted_light"]
    report = f"""# v0.5.1 full frozen-representation extraction

- Source rows: {summary['source_row_count']:,}
- Device: {summary['device_name']}
- SMI-TED: {smi['successful']:,} successful, {smi['unsuccessful']:,} unsuccessful; shape {smi['feature_shape']}
- SMI-TED statuses: {smi['status_counts']}
- Uni-Mol: {uni['successful']:,} successful, {uni['unsuccessful']:,} unsuccessful; shape {uni['feature_shape']}
- Uni-Mol statuses: {uni['status_counts']}
- Repaired deuterium rows: {len(target_indices)}
- Repair reproducibility max |difference|: {reproducibility['maximum_absolute_difference']:.9g} (tolerance {UNIMOL_ATOL:.1g})

Rows remain aligned to the corrected EGFR population. Failed rows are all-NaN. No common-population intersection was computed.

Uni-Mol uses a no-H representation. All hydrogen atoms, including hydrogen isotopes, are removed before inference; deuterium identity is therefore not represented. This is a representation limitation, not a curation change, and the original canonical structures remain unchanged.
"""
    (args.results_dir / "report.md").write_text(report, encoding="utf-8")
    shutil.copyfile(
        args.production_dir / "extraction_audit.parquet",
        args.results_dir / "unimol_v1_extraction_audit.parquet",
    )
    return summary


def repair(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production repair; CPU fallback prohibited")
    print("[3/6] Establishing baseline and re-extracting the 31 affected rows...", flush=True)
    audit_path = args.production_dir / "extraction_audit.parquet"
    audit = pd.read_parquet(audit_path)
    target_mask = audit.status.eq("invalid_coordinates") & audit.failure_reason.eq(
        "heavy_atom_alignment_failed"
    )
    target_indices = audit.index[target_mask].astype(int).tolist()
    if len(target_indices) != TARGET_COUNT:
        raise ValueError(f"Expected exactly 31 repair rows, found {len(target_indices)}")
    if target_indices != audit.loc[target_mask, "original_index"].astype(int).tolist():
        raise ValueError("Audit DataFrame index is not the documented source-row index")

    source = pd.read_parquet(args.dataset, columns=["canonical_smiles"])
    ids = make_molecule_ids(source.canonical_smiles.astype(str))
    stored_ids = np.load(args.production_dir / "molecule_ids.npy")
    if not np.array_equal(ids, stored_ids):
        raise ValueError("Production molecule IDs do not align with corrected EGFR")
    if not np.array_equal(
        source.canonical_smiles.astype(str).to_numpy(), audit.canonical_smiles.to_numpy()
    ):
        raise ValueError("Production audit SMILES do not align with corrected EGFR")

    before_hashes = _backup_production(args.production_dir, args.backup_dir)
    audit_before = audit.copy(deep=True)
    device = torch.device("cuda:0")
    model, dictionary = _load_unimol(args.checkpoint_dir, device)
    repaired_vectors: dict[int, np.ndarray] = {}
    repaired_statuses: dict[int, tuple[str, int | None]] = {}
    for completed, index in enumerate(target_indices, start=1):
        status, reason, mmff_code, molecule, coordinates = generate_unimol_geometry(
            audit.at[index, "canonical_smiles"]
        )
        if status not in SUCCESSFUL_UNIMOL_STATUSES or molecule is None or coordinates is None:
            raise RuntimeError(f"Corrected geometry failed at row {index}: {status}:{reason}")
        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        if any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()):
            raise RuntimeError(f"Hydrogen remained in corrected row {index}")
        vector = _unimol_batch_embeddings(
            model, dictionary,
            [{"index": index, "atoms": atoms, "coordinates": coordinates}], device,
        )[0]
        if vector.shape != (512,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise RuntimeError(f"Invalid repaired embedding at row {index}")
        repaired_vectors[index] = vector
        repaired_statuses[index] = (status, mmff_code)
        if completed % 5 == 0 or completed == TARGET_COUNT:
            print(f"Uni-Mol deuterium repair: {completed}/{TARGET_COUNT}", flush=True)

    # Commit only after all 31 geometries and model forwards have succeeded.
    embeddings = np.load(args.production_dir / "embeddings.npy", mmap_mode="r+")
    success_mask = np.load(args.production_dir / "success_mask.npy")
    for index in target_indices:
        embeddings[index] = repaired_vectors[index]
        success_mask[index] = True
        status, mmff_code = repaired_statuses[index]
        audit.at[index, "status"] = status
        audit.at[index, "failure_reason"] = ""
        audit.at[index, "mmff_code"] = mmff_code
        audit.at[index, "conformer_seed"] = CONFORMER_SEED
    embeddings.flush()
    np.save(args.production_dir / "success_mask.npy", success_mask.astype(bool))
    _write_audit(audit, audit_path)
    del embeddings, model
    torch.cuda.empty_cache()

    print("[4/6] Verifying untouched rows and production alignment...", flush=True)
    _assert_unchanged_rows(args.production_dir, args.backup_dir, np.asarray(target_indices))
    audit_after = pd.read_parquet(audit_path)
    if not audit_after.loc[target_indices, "status"].isin(SUCCESSFUL_UNIMOL_STATUSES).all():
        raise RuntimeError("A repaired row does not have a successful status")
    if not audit_after.loc[target_indices, "failure_reason"].eq("").all():
        raise RuntimeError("A repaired failure reason was not cleared")

    # Reload the model and regenerate three repaired geometries from source SMILES.
    reproducibility_indices = target_indices[:3]
    repeat_model, repeat_dictionary = _load_unimol(args.checkpoint_dir, device)
    stored = np.load(args.production_dir / "embeddings.npy", mmap_mode="r")
    differences: list[float] = []
    for index in reproducibility_indices:
        status, reason, _, molecule, coordinates = generate_unimol_geometry(
            audit_after.at[index, "canonical_smiles"]
        )
        if status not in SUCCESSFUL_UNIMOL_STATUSES or molecule is None or coordinates is None:
            raise RuntimeError(f"Reproducibility geometry failed at row {index}: {reason}")
        repeated = _unimol_batch_embeddings(
            repeat_model, repeat_dictionary,
            [{"index": index, "atoms": [a.GetSymbol() for a in molecule.GetAtoms()],
              "coordinates": coordinates}], device,
        )[0]
        differences.append(float(np.max(np.abs(stored[index] - repeated))))
    maximum_difference = max(differences)
    if maximum_difference > UNIMOL_ATOL:
        raise RuntimeError(
            f"Repair reproducibility {maximum_difference} exceeds {UNIMOL_ATOL}"
        )
    reproducibility = {
        "original_indices": reproducibility_indices,
        "sample_size": len(reproducibility_indices),
        "maximum_absolute_difference": maximum_difference,
        "absolute_tolerance": UNIMOL_ATOL,
        "passed": True,
        "model_reloaded": True,
        "conformers_regenerated_from_scratch": True,
    }
    del repeat_model, stored
    torch.cuda.empty_cache()

    summary = _update_reports(
        args, audit_after, before_hashes, target_indices, reproducibility
    )
    after_hashes = _hashes(args.production_dir)
    repair_record = {
        "repaired_original_indices": target_indices,
        "row_count": len(target_indices),
        "baseline_directory": str(args.backup_dir),
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "non_target_embeddings_exactly_unchanged_including_nan": True,
        "non_target_success_mask_exactly_unchanged": True,
        "molecule_ids_exactly_unchanged": True,
        "non_target_audit_rows_exactly_unchanged": True,
        "reproducibility": reproducibility,
        "final_unimol_summary": summary["representations"]["unimol_v1"],
    }
    _json_dump(args.results_dir / "deuterium_repair.json", repair_record)
    print(json.dumps(repair_record, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v051/targets/EGFR/clean.parquet"))
    parser.add_argument("--production-dir", type=Path, default=Path("data/processed/v051/representations/unimol_v1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/processed/v051/checkpoints/unimol_v1"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/representation_extraction/v051"))
    parser.add_argument("--backup-dir", type=Path, default=Path("data/processed/v051/representations/unimol_v1_pre_deuterium_repair"))
    return parser.parse_args()


if __name__ == "__main__":
    repair(parse_args())
