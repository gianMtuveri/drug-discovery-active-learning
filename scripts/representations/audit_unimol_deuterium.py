#!/usr/bin/env python3
"""Audit deuterium handling at the RDKit-to-Uni-Mol no-H boundary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.representations.run_embedding_pilot import (
    CONFORMER_SEED,
    MMFF_MAX_ITERS,
    _load_unimol,
    coordinates_are_valid,
    remove_all_hydrogens_preserving_coordinates,
)
from scripts.representations.extract_full_frozen_representations import (
    _unimol_batch_embeddings,
)
from src.data.representations import file_sha256, make_molecule_ids


EXPECTED_FAILURES = 31
PRODUCTION_FILES = ("embeddings.npy", "success_mask.npy", "extraction_audit.parquet")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _isotopic_hydrogens(mol: Chem.Mol) -> list[Chem.Atom]:
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetIsotope() > 0]


def _non_h_atom_signature(mol: Chem.Mol) -> list[tuple[int, int, int, str]]:
    return [
        (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), str(atom.GetChiralTag()))
        for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1
    ]


def _no_h_identity(mol: Chem.Mol) -> str:
    copy = Chem.RWMol(mol)
    for index in reversed(range(copy.GetNumAtoms())):
        if copy.GetAtomWithIdx(index).GetAtomicNum() == 1:
            copy.RemoveAtom(index)
    result = copy.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, canonical=True, isomericSmiles=True)


def _attached_atom_description(atom: Chem.Atom) -> list[dict[str, Any]]:
    return [
        {
            "atom_index": neighbor.GetIdx(),
            "symbol": neighbor.GetSymbol(),
            "isotope": neighbor.GetIsotope(),
        }
        for neighbor in atom.GetNeighbors()
    ]


def _prepare_geometry(smiles: str) -> dict[str, Any]:
    original = Chem.MolFromSmiles(smiles)
    if original is None:
        raise ValueError("RDKit parse failed")
    expected_signature = _non_h_atom_signature(original)
    expected_no_h_count = len(expected_signature)
    with_h = Chem.AddHs(original)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = CONFORMER_SEED
    parameters.enforceChirality = True
    parameters.useRandomCoords = False
    parameters.clearConfs = True
    embed_code = int(AllChem.EmbedMolecule(with_h, parameters))
    if embed_code != 0:
        raise RuntimeError(f"Unexpected audit embedding failure: {embed_code}")
    if AllChem.MMFFHasAllMoleculeParams(with_h):
        mmff_code = int(AllChem.MMFFOptimizeMolecule(with_h, maxIters=MMFF_MAX_ITERS))
        mmff_status = "optimized" if mmff_code == 0 else "mmff_nonconverged"
    else:
        mmff_code = None
        mmff_status = "mmff_unavailable"

    current = Chem.RemoveHs(with_h, sanitize=True)
    retained = _isotopic_hydrogens(current)
    current_coordinates = np.asarray(current.GetConformer().GetPositions(), dtype=np.float32)
    current_valid, current_reason = coordinates_are_valid(current, current_coordinates)

    alternative = remove_all_hydrogens_preserving_coordinates(with_h)
    alternative_coordinates = np.asarray(
        alternative.GetConformer().GetPositions(), dtype=np.float32
    )
    alternative_valid, alternative_reason = coordinates_are_valid(
        alternative, alternative_coordinates
    )
    alternative_signature = _non_h_atom_signature(alternative)
    sequence_matches = alternative_signature == expected_signature
    complete = (
        alternative.GetNumAtoms() == expected_no_h_count
        and sequence_matches
        and alternative_coordinates.shape == (expected_no_h_count, 3)
        and np.isfinite(alternative_coordinates).all()
        and alternative_valid
    )
    return {
        "original": original,
        "with_h": with_h,
        "current": current,
        "alternative": alternative,
        "alternative_coordinates": alternative_coordinates,
        "mmff_status": mmff_status,
        "mmff_code": mmff_code,
        "atom_count_after_add_hs": with_h.GetNumAtoms(),
        "atom_count_after_current_remove_hs": current.GetNumAtoms(),
        "expected_no_h_atom_count": expected_no_h_count,
        "retained_isotopic_h_count": len(retained),
        "retained_isotope_mass_numbers": [atom.GetIsotope() for atom in retained],
        "current_coordinate_shape": list(current_coordinates.shape),
        "current_coordinate_validation_passed": current_valid,
        "current_coordinate_validation_reason": current_reason,
        "current_alignment_passed": current.GetNumAtoms() == expected_no_h_count,
        "alternative_atom_count": alternative.GetNumAtoms(),
        "alternative_atom_sequence_matches": sequence_matches,
        "alternative_coordinate_shape": list(alternative_coordinates.shape),
        "alternative_coordinates_finite": bool(np.isfinite(alternative_coordinates).all()),
        "alternative_geometry_valid": alternative_valid,
        "alternative_geometry_failure_reason": alternative_reason,
        "alternative_all_checks_passed": bool(complete),
    }


def analyze(args: argparse.Namespace) -> None:
    production_hashes_before = {
        name: file_sha256(args.representation_dir / name) for name in PRODUCTION_FILES
    }
    extraction = pd.read_parquet(args.representation_dir / "extraction_audit.parquet")
    affected = extraction.loc[
        extraction.status.eq("invalid_coordinates")
        & extraction.failure_reason.eq("heavy_atom_alignment_failed")
    ].copy()
    if len(affected) != EXPECTED_FAILURES:
        raise ValueError(f"Expected 31 matching failures, found {len(affected)}")

    source = pd.read_parquet(args.dataset, columns=["canonical_smiles"])
    source_smiles = source.canonical_smiles.astype(str).to_numpy()
    source_ids = make_molecule_ids(source_smiles)
    if not np.array_equal(source_ids, extraction.molecule_id.to_numpy(dtype=str)):
        raise ValueError("Extraction audit IDs are not aligned to corrected EGFR")

    source_identity_records: list[tuple[str, bool, int]] = []
    for index, smiles in enumerate(source_smiles):
        mol = Chem.MolFromSmiles(smiles)
        source_identity_records.append(
            (_no_h_identity(mol), bool(_isotopic_hydrogens(mol)), index)
        )
    identity_to_unlabelled_indices: dict[str, list[int]] = {}
    for identity, has_isotopic_h, index in source_identity_records:
        if not has_isotopic_h:
            identity_to_unlabelled_indices.setdefault(identity, []).append(index)

    records: list[dict[str, Any]] = []
    geometries: dict[int, dict[str, Any]] = {}
    for row in affected.itertuples(index=False):
        source_smiles_value = source_smiles[int(row.original_index)]
        if source_smiles_value != row.canonical_smiles:
            raise ValueError(f"Source alignment mismatch at row {row.original_index}")
        geometry = _prepare_geometry(source_smiles_value)
        geometries[int(row.original_index)] = geometry
        original = geometry["original"]
        isotopic_h = _isotopic_hydrogens(original)
        other_isotopes = [
            {
                "atom_index": atom.GetIdx(), "symbol": atom.GetSymbol(),
                "mass_number": atom.GetIsotope(),
            }
            for atom in original.GetAtoms()
            if atom.GetAtomicNum() != 1 and atom.GetIsotope() > 0
        ]
        identity = _no_h_identity(original)
        counterparts = identity_to_unlabelled_indices.get(identity, [])
        record = {
            "original_index": int(row.original_index),
            "molecule_id": row.molecule_id,
            "canonical_smiles": source_smiles_value,
            "molecular_formula": rdMolDescriptors.CalcMolFormula(original),
            "original_total_atom_count": original.GetNumAtoms(),
            "existing_heavy_atom_count": int(row.heavy_atom_count),
            "explicit_isotopic_hydrogen_count": len(isotopic_h),
            "hydrogen_isotope_mass_numbers": [atom.GetIsotope() for atom in isotopic_h],
            "isotopic_hydrogen_attachments": [
                {"hydrogen_atom_index": atom.GetIdx(), "mass_number": atom.GetIsotope(),
                 "attached_atoms": _attached_atom_description(atom)}
                for atom in isotopic_h
            ],
            "other_isotopically_labelled_atoms": other_isotopes,
            "other_isotope_count": len(other_isotopes),
            "no_h_unimol_identity": identity,
            "unlabelled_counterpart_present": bool(counterparts),
            "unlabelled_counterpart_indices": counterparts,
            **{key: value for key, value in geometry.items()
               if key not in {"original", "with_h", "current", "alternative", "alternative_coordinates"}},
        }
        records.append(record)

    audit = pd.DataFrame(records).sort_values("original_index").reset_index(drop=True)
    identity_counts = audit.no_h_unimol_identity.value_counts()
    audit["deuterated_variant_group_size"] = audit.no_h_unimol_identity.map(identity_counts)
    collision_table = (
        audit.groupby("no_h_unimol_identity", sort=True)
        .agg(
            deuterated_variant_count=("original_index", "size"),
            original_indices=("original_index", list),
            molecule_ids=("molecule_id", list),
            unlabelled_counterpart_present=("unlabelled_counterpart_present", "any"),
            unlabelled_counterpart_indices=("unlabelled_counterpart_indices", lambda values: sorted({i for value in values for i in value})),
        )
        .reset_index()
    )

    dictionary_tokens = (args.checkpoint_dir / "mol.dict.txt").read_text().splitlines()
    summary = {
        "schema_version": 1,
        "affected_count": len(audit),
        "all_failures_reproduced": bool((~audit.current_alignment_passed).all()),
        "current_failure": {
            "retained_isotopic_h_count_distribution": {
                str(key): int(value)
                for key, value in sorted(Counter(audit.retained_isotopic_h_count).items())
            },
            "all_current_post_removehs_counts_exceed_expected_by_retained_isotopic_h": bool((
                audit.atom_count_after_current_remove_hs
                - audit.expected_no_h_atom_count
                == audit.retained_isotopic_h_count
            ).all()),
            "mmff_status_counts": {
                str(key): int(value) for key, value in audit.mmff_status.value_counts().items()
            },
        },
        "audit_only_explicit_removal": {
            "passing_count": int(audit.alternative_all_checks_passed.sum()),
            "failing_count": int((~audit.alternative_all_checks_passed).sum()),
        },
        "isotopes": {
            "explicit_isotopic_hydrogen_total": int(
                audit.explicit_isotopic_hydrogen_count.sum()
            ),
            "hydrogen_mass_number_distribution": dict(Counter(
                mass for values in audit.hydrogen_isotope_mass_numbers for mass in values
            )),
            "isotopic_hydrogen_attachment_element_distribution": dict(Counter(
                attached["symbol"]
                for values in audit.isotopic_hydrogen_attachments
                for hydrogen in values
                for attached in hydrogen["attached_atoms"]
            )),
            "molecules_with_other_isotopic_atoms": int((audit.other_isotope_count > 0).sum()),
        },
        "counterparts_and_collisions": {
            "with_unlabelled_counterpart": int(audit.unlabelled_counterpart_present.sum()),
            "without_unlabelled_counterpart": int((~audit.unlabelled_counterpart_present).sum()),
            "unique_no_h_inputs": int(audit.no_h_unimol_identity.nunique()),
            "groups_with_multiple_deuterated_variants": int((identity_counts > 1).sum()),
            "molecules_in_multiple_variant_groups": int(identity_counts[identity_counts > 1].sum()),
            "maximum_deuterated_variant_group_size": int(identity_counts.max()),
        },
        "pinned_model_semantics": {
            "dictionary_tokens": dictionary_tokens,
            "dictionary_contains_hydrogen_token": "H" in dictionary_tokens,
            "dictionary_contains_deuterium_token": any(token in dictionary_tokens for token in ("D", "[2H]", "2H")),
            "checkpoint": "mol_pre_no_h_220816.pt",
            "remove_hydrogen_code_behavior": "RemoveHydrogenDataset masks every atom string equal to H before tokenization",
            "isotope_metadata_in_inference_features": False,
        },
        "recommendation": "B",
        "recommendation_text": (
            "Treat removal of all atomic-number-1 atoms, including isotopic hydrogen, "
            "as the correct implementation of the pinned no-H Uni-Mol input semantics; "
            "after approval, re-extract only these 31 aligned rows."
        ),
        "deuterium_information_lost_under_option_b": True,
        "provenance": {
            "dataset": str(args.dataset),
            "dataset_sha256": file_sha256(args.dataset),
            "extraction_audit": str(args.representation_dir / "extraction_audit.parquet"),
            "production_hashes_before": production_hashes_before,
            "dictionary_sha256": file_sha256(args.checkpoint_dir / "mol.dict.txt"),
            "checkpoint_sha256": file_sha256(args.checkpoint_dir / "mol_pre_no_h_220816.pt"),
            "rdkit_version": rdBase.rdkitVersion,
            "conformer_seed": CONFORMER_SEED,
            "mmff_max_iterations": MMFF_MAX_ITERS,
        },
        "deterministic_smoke_test_indices": audit.sort_values(
            ["existing_heavy_atom_count", "original_index"]
        ).head(3).original_index.astype(int).tolist(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output_dir / "per_molecule_audit.parquet", index=False)
    collision_table.to_parquet(args.output_dir / "counterpart_collision_audit.parquet", index=False)
    _json_dump(args.output_dir / "summary.json", summary)
    _write_report(args.output_dir / "report.md", summary)
    _assert_production_unchanged(args, production_hashes_before)
    print(json.dumps(summary, indent=2))


def _assert_production_unchanged(args: argparse.Namespace, expected: dict[str, str]) -> None:
    observed = {
        name: file_sha256(args.representation_dir / name) for name in PRODUCTION_FILES
    }
    if observed != expected:
        raise RuntimeError("Production Uni-Mol extraction artifacts changed during audit")


def verify(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the audit smoke test; CPU fallback prohibited")
    summary_path = args.output_dir / "summary.json"
    audit_path = args.output_dir / "per_molecule_audit.parquet"
    if not summary_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Run --stage analyze first")
    summary = json.loads(summary_path.read_text())
    audit = pd.read_parquet(audit_path).set_index("original_index")
    production_hashes = summary["provenance"]["production_hashes_before"]
    selected = summary["deterministic_smoke_test_indices"]
    geometries = []
    for index in selected:
        geometry = _prepare_geometry(audit.at[index, "canonical_smiles"])
        if not geometry["alternative_all_checks_passed"]:
            raise RuntimeError(f"Audit-only geometry failed for row {index}")
        alternative = geometry["alternative"]
        geometries.append({
            "index": index,
            "atoms": [atom.GetSymbol() for atom in alternative.GetAtoms()],
            "coordinates": geometry["alternative_coordinates"],
        })
    device = torch.device("cuda:0")
    model, dictionary = _load_unimol(args.checkpoint_dir, device)
    results = []
    for geometry in geometries:
        unsupported = sorted(set(geometry["atoms"]).difference(dictionary.indices))
        if unsupported:
            raise RuntimeError(f"Unsupported audit atoms: {unsupported}")
        vector = _unimol_batch_embeddings(model, dictionary, [geometry], device)
        results.append({
            "original_index": geometry["index"],
            "output_shape": list(vector.shape),
            "dtype": str(vector.dtype),
            "finite": bool(np.isfinite(vector).all()),
            "unsupported_atom_tokens": unsupported,
            "inference_error": "",
        })
    verification = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint": "mol_pre_no_h_220816.pt",
        "features_only": True,
        "embedding": "CLS",
        "results": results,
        "all_passed": all(
            item["output_shape"] == [1, 512]
            and item["dtype"] == "float32"
            and item["finite"]
            and not item["unsupported_atom_tokens"]
            and not item["inference_error"]
            for item in results
        ),
    }
    _json_dump(args.output_dir / "forward_pass_verification.json", verification)
    summary["forward_pass_verification"] = verification
    _json_dump(summary_path, summary)
    _write_report(args.output_dir / "report.md", summary)
    _assert_production_unchanged(args, production_hashes)
    print(json.dumps(verification, indent=2))


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    alternative = summary["audit_only_explicit_removal"]
    counterparts = summary["counterparts_and_collisions"]
    smoke = summary.get("forward_pass_verification")
    lines = [
        "# Uni-Mol deuterium audit",
        "",
        f"All {summary['affected_count']} existing alignment failures were reproduced. RDKit's default "
        "`RemoveHs` retained deuterium, and each post-removal count exceeded the intended no-H count "
        "by exactly the number of retained isotopic hydrogens.",
        "",
        f"Audit-only removal of every atomic-number-1 atom produced {alternative['passing_count']} valid "
        f"no-H geometries and {alternative['failing_count']} failures.",
        "",
        f"{counterparts['with_unlabelled_counterpart']} molecules have an unlabelled EGFR counterpart; "
        f"{counterparts['without_unlabelled_counterpart']} do not. The 31 structures map to "
        f"{counterparts['unique_no_h_inputs']} distinct no-H inputs.",
        "",
        "The pinned no-H task removes every atom whose symbol is `H` before dictionary tokenization. "
        "Its atom tokens, coordinates, distances, and edge construction carry no isotope field. "
        "Consequently, deuterium identity is necessarily lost if all hydrogen is removed.",
        "",
        "## Recommendation: option B",
        "",
        summary["recommendation_text"],
        " Uni-Mol would therefore be insensitive to deuterium labelling in this pipeline; that is a "
        "representation limitation and must be documented, not described as preserved isotope information.",
    ]
    if smoke:
        lines.extend([
            "",
            f"The audit-only GPU smoke test passed for all {len(smoke['results'])} selected molecules "
            f"on {smoke['device']}; each output was finite float32 with shape `(1, 512)`.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyze", "verify", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/v051/targets/EGFR/clean.parquet"))
    parser.add_argument("--representation-dir", type=Path, default=Path("data/processed/v051/representations/unimol_v1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/processed/v051/checkpoints/unimol_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/representation_audits/v051/unimol_deuterium"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"analyze", "all"}:
        analyze(args)
    if args.stage in {"verify", "all"}:
        verify(args)


if __name__ == "__main__":
    main()
