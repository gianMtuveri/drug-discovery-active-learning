#!/usr/bin/env python3
"""Deterministic v0.5.1 SMI-TED Light and Uni-Mol v1 pilot.

The prepare stage uses molecular structure only.  The inference stage requires
CUDA explicitly and never falls back to CPU.  Failed molecules retain aligned
rows and explicit status metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
import torch

from src.data.representations import file_sha256, make_molecule_ids, ordered_population_hash


PILOT_SCHEMA_VERSION = 1
PILOT_SIZE = 20
CONFORMER_SEED = 0xC0FFEE
MMFF_MAX_ITERS = 1000
SMI_TED_MAX_TOTAL_TOKENS = 202
UNIMOL_MAX_ATOMS = 256


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass
class PilotRecord:
    original_index: int
    molecule_id: str
    canonical_smiles: str
    smi_ted_normalized_smiles: str | None
    heavy_atom_count: int | None
    smi_ted_token_count: int | None
    has_stereochemistry: bool
    selection_reason: str
    smi_ted_status: str
    smi_ted_failure_reason: str
    unimol_status: str
    unimol_failure_reason: str
    mmff_code: int | None = None
    conformer_seed: int | None = None


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_validation_report(path: Path, summary: dict[str, Any], provenance: dict[str, Any]) -> None:
    repro = summary["reproducibility"]
    lines = [
        "# v0.5.1 embedding pilot validation",
        "",
        f"- Pilot population: {summary['pilot_size']} molecules",
        f"- Heavy-atom range: {summary['heavy_atom_range'][0]}–{summary['heavy_atom_range'][1]}",
        f"- SMI-TED total-token range: {summary['smi_ted_token_range'][0]}–{summary['smi_ted_token_range'][1]}",
        f"- Device: {provenance['device_name']} ({provenance['device']})",
        f"- SMI-TED aligned array: {summary['smi_ted_embedding_shape']}; successful rows: {summary['successful_smi_ted_rows']}",
        f"- Uni-Mol aligned array: {summary['unimol_embedding_shape']}; successful rows: {summary['successful_unimol_rows']}",
        f"- SMI-TED statuses: {summary['smi_ted_status_counts']}",
        f"- Uni-Mol statuses: {summary['unimol_status_counts']}",
        f"- SMI-TED reproducibility maximum absolute difference: {repro['smi_ted']['maximum_absolute_difference']:.9g} "
        f"(absolute tolerance {repro['smi_ted']['atol']:.9g})",
        f"- Uni-Mol reproducibility maximum absolute difference: {repro['unimol']['maximum_absolute_difference']:.9g} "
        f"(absolute tolerance {repro['unimol']['atol']:.9g})",
        "",
        "Inputs over 202 SMI-TED tokens or 256 heavy atoms were rejected before model inference. "
        "Uni-Mol geometries are single deterministic ETKDGv3 conformers; they are computational conformations, not experimental bioactive poses.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_smi_module(checkpoint_dir: Path):
    sys.path.insert(0, str(checkpoint_dir))
    spec = importlib.util.spec_from_file_location(
        "official_smi_ted_light_load", checkpoint_dir / "load.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the official SMI-TED loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_stereochemistry(mol: Chem.Mol, smiles: str) -> bool:
    assigned = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    return bool(assigned) or "@" in smiles or "/" in smiles or "\\" in smiles


def _nearest_unused(frame: pd.DataFrame, column: str, target: float, used: set[int]) -> int:
    candidates = frame.loc[~frame.index.isin(used), [column, "original_index"]].copy()
    candidates["distance"] = (candidates[column] - target).abs()
    return int(candidates.sort_values(["distance", "original_index"]).iloc[0]["original_index"])


def select_pilot(frame: pd.DataFrame, pilot_size: int = PILOT_SIZE) -> dict[int, str]:
    """Select structural extremes and quantiles with deterministic tie-breaking."""
    required = {
        "original_index", "heavy_atom_count", "smi_ted_token_count",
        "smiles_length", "has_stereochemistry",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pilot selection is missing columns: {sorted(missing)}")

    chosen: dict[int, str] = {}

    def add(index: int, reason: str) -> None:
        chosen.setdefault(int(index), reason)

    ordered = frame.sort_values("original_index")
    add(int(frame.sort_values(["heavy_atom_count", "original_index"]).iloc[0].original_index), "minimum_heavy_atoms")
    add(int(frame.sort_values(["heavy_atom_count", "original_index"], ascending=[False, True]).iloc[0].original_index), "maximum_heavy_atoms")
    eligible = frame.loc[frame.smi_ted_token_count <= SMI_TED_MAX_TOTAL_TOKENS]
    add(int(eligible.sort_values(["smi_ted_token_count", "original_index"], ascending=[False, True]).iloc[0].original_index), "nearest_smi_ted_limit")
    add(int(eligible.sort_values(["heavy_atom_count", "original_index"], ascending=[False, True]).iloc[0].original_index), "largest_unimol_eligible")
    nonstereo = frame.loc[~frame.has_stereochemistry]
    add(int(nonstereo.sort_values(["smi_ted_token_count", "original_index"], ascending=[False, True]).iloc[0].original_index), "longest_nonstereochemical")

    for quantile in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        target = float(frame.heavy_atom_count.quantile(quantile))
        add(_nearest_unused(frame, "heavy_atom_count", target, set(chosen)), f"heavy_atom_quantile_{quantile:.2f}")

    for quantile in (0.25, 0.50, 0.75, 0.90, 0.99):
        target = float(frame.smi_ted_token_count.quantile(quantile))
        add(_nearest_unused(frame, "smi_ted_token_count", target, set(chosen)), f"token_quantile_{quantile:.2f}")

    stereo = frame.loc[frame.has_stereochemistry]
    for label, quantile in (("small", 0.10), ("median", 0.50), ("large", 0.90)):
        target = float(stereo.heavy_atom_count.quantile(quantile))
        add(_nearest_unused(stereo, "heavy_atom_count", target, set(chosen)), f"stereochemical_{label}")

    # Deterministic farthest-point completion in structural-property space.
    numeric = frame[["heavy_atom_count", "smi_ted_token_count", "smiles_length"]].astype(float)
    scales = numeric.max() - numeric.min()
    scales[scales == 0] = 1.0
    points = ((numeric - numeric.min()) / scales).to_numpy()
    stereo_axis = frame.has_stereochemistry.astype(float).to_numpy()[:, None]
    points = np.concatenate([points, stereo_axis], axis=1)
    index_to_position = {int(v): i for i, v in enumerate(frame.original_index)}
    while len(chosen) < pilot_size:
        selected_positions = [index_to_position[i] for i in chosen]
        distances = np.linalg.norm(points[:, None, :] - points[selected_positions][None, :, :], axis=2)
        nearest = distances.min(axis=1)
        for index in chosen:
            nearest[index_to_position[index]] = -1.0
        best_distance = nearest.max()
        tied_positions = np.flatnonzero(np.isclose(nearest, best_distance))
        best = min(int(frame.iloc[pos].original_index) for pos in tied_positions)
        add(best, "structural_coverage_fill")

    if len(chosen) > pilot_size:
        chosen = dict(list(chosen.items())[:pilot_size])
    return chosen


def coordinates_are_valid(mol: Chem.Mol, coordinates: np.ndarray) -> tuple[bool, str]:
    if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
        return False, "conformer_not_marked_3d"
    if coordinates.shape != (mol.GetNumAtoms(), 3):
        return False, "coordinate_atom_count_mismatch"
    if not np.isfinite(coordinates).all():
        return False, "non_finite_coordinates"
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) <= 1e-6:
        return False, "zero_or_collapsed_coordinates"
    if mol.GetNumAtoms() > 1:
        distances = np.linalg.norm(centered[:, None, :] - centered[None, :, :], axis=-1)
        if float(distances.max()) <= 1e-3:
            return False, "zero_or_collapsed_coordinates"
    return True, ""


def remove_all_hydrogens_preserving_coordinates(mol: Chem.Mol) -> Chem.Mol:
    """Build the Uni-Mol no-H input by removing every atomic-number-1 atom.

    RDKit's default ``RemoveHs`` retains isotope-labelled hydrogen.  Uni-Mol's
    pinned no-H preprocessing removes atoms by element symbol, so deuterium and
    other hydrogen isotopes must also be absent from the temporary model input.
    Non-hydrogen atom ordering and conformer coordinates are preserved.
    """
    editable = Chem.RWMol(mol)
    for atom_index in reversed(range(editable.GetNumAtoms())):
        if editable.GetAtomWithIdx(atom_index).GetAtomicNum() == 1:
            editable.RemoveAtom(atom_index)
    no_h = editable.GetMol()
    Chem.SanitizeMol(no_h)
    return no_h


def generate_unimol_geometry(smiles: str) -> tuple[str, str, int | None, Chem.Mol | None, np.ndarray | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "embedding_failed", "rdkit_parse_failed", None, None, None
    heavy_count = mol.GetNumHeavyAtoms()
    if heavy_count > UNIMOL_MAX_ATOMS:
        return "atom_limit_exceeded", f"{heavy_count}>{UNIMOL_MAX_ATOMS}", None, None, None

    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = CONFORMER_SEED
    params.enforceChirality = True
    params.useRandomCoords = False
    params.clearConfs = True
    embed_code = AllChem.EmbedMolecule(with_h, params)
    if embed_code != 0 or with_h.GetNumConformers() != 1:
        return "embedding_failed", f"rdkit_embed_code={embed_code}", None, None, None

    if not AllChem.MMFFHasAllMoleculeParams(with_h):
        status, reason, mmff_code = "mmff_unavailable", "mmff_parameters_unavailable", None
    else:
        try:
            mmff_code = int(AllChem.MMFFOptimizeMolecule(with_h, maxIters=MMFF_MAX_ITERS))
            if mmff_code == 0:
                status, reason = "optimized", ""
            elif mmff_code == 1:
                status, reason = "mmff_nonconverged", "mmff_max_iterations_reached"
            else:
                status, reason = "mmff_unavailable", f"mmff_return_code={mmff_code}"
        except Exception as exc:  # RDKit raises for unsupported force-field cases.
            status, reason, mmff_code = "mmff_unavailable", f"mmff_exception:{type(exc).__name__}", None

    heavy = remove_all_hydrogens_preserving_coordinates(with_h)
    if heavy.GetNumAtoms() != heavy_count or heavy.GetNumConformers() != 1:
        return "invalid_coordinates", "heavy_atom_alignment_failed", mmff_code, None, None
    coordinates = np.asarray(heavy.GetConformer().GetPositions(), dtype=np.float32)
    valid, validation_reason = coordinates_are_valid(heavy, coordinates)
    if not valid:
        return "invalid_coordinates", validation_reason, mmff_code, None, None
    return status, reason, mmff_code, heavy, coordinates


def _population_frame(dataset_path: Path, smi_module, vocab_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(dataset_path, columns=["canonical_smiles"])
    smiles = source.canonical_smiles.astype(str).to_numpy()
    ids = make_molecule_ids(smiles)
    tokenizer = smi_module.MolTranBertTokenizer(str(vocab_path))
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Corrected EGFR canonical SMILES failed parsing at row {index}")
        normalized = smi_module.normalize_smiles(value)
        if normalized is None:
            token_count = None
        else:
            token_count = len(tokenizer(normalized, add_special_tokens=True, truncation=False)["input_ids"])
        rows.append({
            "original_index": index,
            "molecule_id": str(ids[index]),
            "canonical_smiles": value,
            "smi_ted_normalized_smiles": normalized,
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "smi_ted_token_count": token_count,
            "smiles_length": len(value),
            "has_stereochemistry": _has_stereochemistry(mol, value),
        })
    return pd.DataFrame(rows)


def prepare(args: argparse.Namespace) -> None:
    smi_dir = args.checkpoint_dir / "smi_ted_light"
    smi_module = _load_smi_module(smi_dir)
    population = _population_frame(args.dataset, smi_module, smi_dir / "bert_vocab_curated.txt")
    selected = select_pilot(population, args.pilot_size)
    pilot = population.loc[population.original_index.isin(selected)].copy()
    pilot["selection_reason"] = pilot.original_index.map(selected)
    pilot = pilot.sort_values("original_index").reset_index(drop=True)

    records: list[PilotRecord] = []
    geometry_dir = args.output_dir / "unimol_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for row in pilot.itertuples(index=False):
        if row.smi_ted_normalized_smiles is None:
            smi_status, smi_reason = "normalization_failed", "official_normalization_failed"
        elif int(row.smi_ted_token_count) > SMI_TED_MAX_TOTAL_TOKENS:
            smi_status = "token_limit_exceeded"
            smi_reason = f"{int(row.smi_ted_token_count)}>{SMI_TED_MAX_TOTAL_TOKENS}"
        else:
            smi_status, smi_reason = "eligible", ""

        uni_status, uni_reason, mmff_code, heavy, coordinates = generate_unimol_geometry(row.canonical_smiles)
        if heavy is not None and coordinates is not None:
            atoms = np.asarray([atom.GetSymbol() for atom in heavy.GetAtoms()], dtype="U3")
            np.savez_compressed(
                geometry_dir / f"{row.molecule_id}.npz",
                atoms=atoms,
                coordinates=coordinates,
            )
        records.append(PilotRecord(
            original_index=int(row.original_index), molecule_id=row.molecule_id,
            canonical_smiles=row.canonical_smiles,
            smi_ted_normalized_smiles=row.smi_ted_normalized_smiles,
            heavy_atom_count=int(row.heavy_atom_count),
            smi_ted_token_count=int(row.smi_ted_token_count) if row.smi_ted_token_count is not None else None,
            has_stereochemistry=bool(row.has_stereochemistry), selection_reason=row.selection_reason,
            smi_ted_status=smi_status, smi_ted_failure_reason=smi_reason,
            unimol_status=uni_status, unimol_failure_reason=uni_reason,
            mmff_code=mmff_code, conformer_seed=None if uni_status == "atom_limit_exceeded" else CONFORMER_SEED,
        ))

    audit = pd.DataFrame([asdict(record) for record in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "pilot_molecule_audit.csv", index=False)
    audit.to_json(args.output_dir / "pilot_molecule_audit.json", orient="records", indent=2)
    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "selection_policy": "structure_only_extremes_quantiles_stereochemistry_and_farthest_fill",
        "selection_uses_assay_fields": False,
        "source_dataset": str(args.dataset),
        "source_dataset_sha256": file_sha256(args.dataset),
        "source_population_size": len(population),
        "source_population_hash": ordered_population_hash(make_molecule_ids(population.canonical_smiles)),
        "pilot_size": len(audit),
        "ordered_original_indices": audit.original_index.astype(int).tolist(),
        "ordered_molecule_ids": audit.molecule_id.tolist(),
        "smi_ted_max_total_tokens": SMI_TED_MAX_TOTAL_TOKENS,
        "unimol_max_atoms_before_special_tokens": UNIMOL_MAX_ATOMS,
        "unimol_max_total_tokens": UNIMOL_MAX_ATOMS + 2,
        "conformer": {
            "rdkit_version": rdBase.rdkitVersion,
            "method": "ETKDGv3",
            "seed": CONFORMER_SEED,
            "add_hydrogens_for_generation": True,
            "remove_hydrogens_for_inference": True,
            "mmff_max_iterations": MMFF_MAX_ITERS,
            "fallback_coordinates": False,
        },
    }
    _json_dump(args.output_dir / "pilot_population_manifest.json", manifest)
    print(json.dumps({
        "pilot_size": len(audit),
        "indices": manifest["ordered_original_indices"],
        "smi_status": audit.smi_ted_status.value_counts().to_dict(),
        "unimol_status": audit.unimol_status.value_counts().to_dict(),
    }, indent=2))


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real pilot; CPU fallback is prohibited")
    return torch.device("cuda:0")


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _load_unimol(checkpoint_dir: Path, device: torch.device):
    from argparse import Namespace
    from unicore.data import Dictionary
    from unimol.models.unimol import UniMolModel

    dictionary = Dictionary.load(str(checkpoint_dir / "mol.dict.txt"))
    dictionary.add_symbol("[MASK]", is_special=True)
    model_args = Namespace(
        mode="infer", masked_token_loss=1.0, masked_coord_loss=5.0,
        masked_dist_loss=10.0, x_norm_loss=0.01,
        delta_pair_repr_norm_loss=0.01,
    )
    model = UniMolModel(model_args, dictionary)
    state = torch.load(checkpoint_dir / "mol_pre_no_h_220816.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    _freeze(model)
    return model.to(device), dictionary


def _unimol_embedding(model, dictionary, geometry_path: Path, device: torch.device) -> np.ndarray:
    with np.load(geometry_path, allow_pickle=False) as artifact:
        atoms = artifact["atoms"].astype(str)
        coordinates = artifact["coordinates"].astype(np.float32)
    unknown = sorted(set(atoms).difference(dictionary.indices))
    if unknown:
        raise ValueError(f"Unsupported dictionary atom types: {unknown}")
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    ids = [dictionary.bos(), *[dictionary.index(atom) for atom in atoms], dictionary.eos()]
    coords = np.concatenate([np.zeros((1, 3), np.float32), centered, np.zeros((1, 3), np.float32)])
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    coord_tensor = torch.tensor(coords[None], dtype=torch.float32, device=device)
    distance = torch.cdist(coord_tensor, coord_tensor)
    edge = tokens.unsqueeze(-1) * len(dictionary) + tokens.unsqueeze(-2)
    with torch.inference_mode():
        representation, _ = model(tokens, distance, coord_tensor, edge, features_only=True)
    return representation[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def infer(args: argparse.Namespace) -> None:
    device = _require_cuda()
    audit_path = args.output_dir / "pilot_molecule_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError("Run --stage prepare before inference")
    audit = pd.read_csv(audit_path, keep_default_na=False)
    n = len(audit)
    smi_features = np.full((n, 768), np.nan, dtype=np.float32)
    uni_features = np.full((n, 512), np.nan, dtype=np.float32)
    reproducibility: dict[str, Any] = {}

    smi_dir = args.checkpoint_dir / "smi_ted_light"
    smi_module = _load_smi_module(smi_dir)
    smi_model = smi_module.load_smi_ted(str(smi_dir), "smi-ted-Light_40.pt", "bert_vocab_curated.txt")
    _freeze(smi_model)
    eligible_smi = audit.index[audit.smi_ted_status == "eligible"].tolist()
    smi_inputs = audit.loc[eligible_smi, "canonical_smiles"].tolist()
    with torch.inference_mode():
        encoded = smi_model.encode(smi_inputs, batch_size=min(8, len(smi_inputs)), return_torch=True)
    encoded_np = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
    if encoded_np.shape != (len(eligible_smi), 768) or not np.isfinite(encoded_np).all():
        raise RuntimeError("SMI-TED produced invalid pilot embeddings")
    smi_features[eligible_smi] = encoded_np
    repeat_indices = eligible_smi[: min(3, len(eligible_smi))]
    repeat_inputs = audit.loc[repeat_indices, "canonical_smiles"].tolist()
    del smi_model, encoded
    torch.cuda.empty_cache()
    smi_repeat_model = smi_module.load_smi_ted(str(smi_dir), "smi-ted-Light_40.pt", "bert_vocab_curated.txt")
    _freeze(smi_repeat_model)
    with torch.inference_mode():
        repeated = smi_repeat_model.encode(repeat_inputs, batch_size=1, return_torch=True).cpu().numpy().astype(np.float32)
    reproducibility["smi_ted"] = {
        "row_indices": repeat_indices,
        "atol": 1e-5,
        "maximum_absolute_difference": float(np.max(np.abs(smi_features[repeat_indices] - repeated))),
    }
    reproducibility["smi_ted"]["within_tolerance"] = (
        reproducibility["smi_ted"]["maximum_absolute_difference"]
        <= reproducibility["smi_ted"]["atol"]
    )
    del smi_repeat_model, repeated, smi_module
    torch.cuda.empty_cache()

    uni_dir = args.checkpoint_dir / "unimol_v1"
    uni_model, dictionary = _load_unimol(uni_dir, device)
    geometry_success = {"optimized", "mmff_unavailable", "mmff_nonconverged"}
    eligible_uni: list[int] = []
    for index, row in audit.iterrows():
        if row.unimol_status not in geometry_success:
            continue
        geometry_path = args.output_dir / "unimol_geometry" / f"{row.molecule_id}.npz"
        try:
            vector = _unimol_embedding(uni_model, dictionary, geometry_path, device)
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError("invalid CLS vector")
            uni_features[index] = vector
            eligible_uni.append(index)
        except Exception as exc:
            audit.at[index, "unimol_status"] = "inference_failed"
            audit.at[index, "unimol_failure_reason"] = f"{type(exc).__name__}:{exc}"

    repeat_uni = eligible_uni[: min(3, len(eligible_uni))]
    del uni_model
    torch.cuda.empty_cache()
    uni_repeat_model, repeat_dictionary = _load_unimol(uni_dir, device)
    repeated_uni: list[np.ndarray] = []
    for index in repeat_uni:
        row = audit.iloc[index]
        # Rebuild geometry from the canonical structure, not the cached coordinates.
        status, reason, _, heavy, coordinates = generate_unimol_geometry(row.canonical_smiles)
        if status not in geometry_success or heavy is None or coordinates is None:
            raise RuntimeError(f"Reproducibility conformer failed: {status}:{reason}")
        temporary = args.output_dir / "unimol_geometry" / ".reproducibility_tmp.npz"
        np.savez_compressed(temporary, atoms=np.asarray([a.GetSymbol() for a in heavy.GetAtoms()], dtype="U3"), coordinates=coordinates)
        repeated_uni.append(_unimol_embedding(uni_repeat_model, repeat_dictionary, temporary, device))
        temporary.unlink()
    repeated_uni_array = np.asarray(repeated_uni, dtype=np.float32)
    reproducibility["unimol"] = {
        "row_indices": repeat_uni,
        "atol": 1e-6,
        "maximum_absolute_difference": float(np.max(np.abs(uni_features[repeat_uni] - repeated_uni_array))),
        "geometry_regenerated_from_scratch": True,
    }
    reproducibility["unimol"]["within_tolerance"] = (
        reproducibility["unimol"]["maximum_absolute_difference"]
        <= reproducibility["unimol"]["atol"]
    )

    np.save(args.output_dir / "smi_ted_embeddings.npy", smi_features)
    np.save(args.output_dir / "unimol_embeddings.npy", uni_features)
    audit.to_csv(audit_path, index=False)
    audit.to_json(args.output_dir / "pilot_molecule_audit.json", orient="records", indent=2)
    provenance = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "runtime_versions": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "rdkit", "torch", "transformers", "unicore", "unimol")
        },
        "smi_ted": {
            "checkpoint": "smi-ted-Light_40.pt", "dimension": 768,
            "implementation": "official IBM SMI-TED Light load.py and adapted fast_transformers",
            "checkpoint_sha256": file_sha256(smi_dir / "smi-ted-Light_40.pt"),
            "loader_sha256": file_sha256(smi_dir / "load.py"),
            "vocabulary_sha256": file_sha256(smi_dir / "bert_vocab_curated.txt"),
        },
        "unimol": {
            "checkpoint": "mol_pre_no_h_220816.pt", "dimension": 512,
            "implementation": "original Uni-Mol v1 molecular encoder via Uni-Core",
            "checkpoint_sha256": file_sha256(uni_dir / "mol_pre_no_h_220816.pt"),
            "dictionary_sha256": file_sha256(uni_dir / "mol.dict.txt"),
            "features_only": True,
        },
        "reproducibility": reproducibility,
    }
    _json_dump(args.output_dir / "model_provenance.json", provenance)
    summary = {
        "pilot_size": n,
        "smi_ted_status_counts": audit.smi_ted_status.value_counts().to_dict(),
        "unimol_status_counts": audit.unimol_status.value_counts().to_dict(),
        "smi_ted_embedding_shape": list(smi_features.shape),
        "unimol_embedding_shape": list(uni_features.shape),
        "successful_smi_ted_rows": int(np.isfinite(smi_features).all(axis=1).sum()),
        "successful_unimol_rows": int(np.isfinite(uni_features).all(axis=1).sum()),
        "heavy_atom_range": [int(audit.heavy_atom_count.min()), int(audit.heavy_atom_count.max())],
        "smi_ted_token_range": [int(audit.smi_ted_token_count.min()), int(audit.smi_ted_token_count.max())],
        "reproducibility": reproducibility,
    }
    _json_dump(args.output_dir / "validation_summary.json", summary)
    _write_validation_report(args.output_dir / "validation_report.md", summary, provenance)
    print(json.dumps(summary, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v051/targets/EGFR/clean.parquet"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/processed/v051/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/representation_pilots/v051"))
    parser.add_argument("--pilot-size", type=int, default=PILOT_SIZE)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"infer", "all"}:
        infer(args)


if __name__ == "__main__":
    main()
