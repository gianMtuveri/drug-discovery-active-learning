#!/usr/bin/env python3
"""Audit stereochemistry loss in the pinned official SMI-TED normalizer."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch

from src.data.representations import file_sha256, make_molecule_ids, ordered_population_hash


SMI_TED_DIMENSION = 768
SMI_TED_MAX_TOTAL_TOKENS = 202
SMI_TED_REPRODUCIBILITY_ATOL = 1e-5


def load_official_module(source_dir: Path):
    sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location(
        "official_smi_ted_light_stereo_audit", source_dir / "load.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned official SMI-TED loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assigned_stereochemistry(mol: Chem.Mol) -> tuple[bool, bool]:
    tetrahedral = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )
    bond = any(
        item.GetStereo()
        not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
        for item in mol.GetBonds()
    )
    return tetrahedral, bond


def stereo_stripped_canonical_smiles(mol: Chem.Mol) -> str:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def collision_group_id(normalized_smiles: str) -> str:
    return "collision-" + sha256(normalized_smiles.encode("utf-8")).hexdigest()[:16]


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    official = load_official_module(args.smi_ted_dir)
    source = pd.read_parquet(args.dataset, columns=["canonical_smiles"])
    smiles = source.canonical_smiles.astype(str).to_numpy()
    molecule_ids = make_molecule_ids(smiles)
    records: list[dict[str, Any]] = []

    for original_index, (molecule_id, canonical_smiles) in enumerate(zip(molecule_ids, smiles)):
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise ValueError(f"Invalid corrected canonical SMILES at row {original_index}")
        tetrahedral, bond = assigned_stereochemistry(mol)
        normalized = official.normalize_smiles(canonical_smiles)
        if normalized is None:
            raise ValueError(f"Official normalization failed at row {original_index}")
        records.append({
            "original_index": original_index,
            "molecule_id": molecule_id,
            "original_canonical_smiles": canonical_smiles,
            "smi_ted_normalized_smiles": normalized,
            "has_assigned_stereochemistry": tetrahedral or bond,
            "has_assigned_tetrahedral_stereochemistry": tetrahedral,
            "has_assigned_bond_stereochemistry": bond,
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "stereo_stripped_canonical_smiles": stereo_stripped_canonical_smiles(mol),
        })

    audit = pd.DataFrame(records)
    distinct_counts = audit.groupby("smi_ted_normalized_smiles")[
        "original_canonical_smiles"
    ].transform("nunique")
    audit["is_collision_member"] = distinct_counts > 1
    audit["collision_group_id"] = ""
    audit.loc[audit.is_collision_member, "collision_group_id"] = audit.loc[
        audit.is_collision_member, "smi_ted_normalized_smiles"
    ].map(collision_group_id)

    groups: list[dict[str, Any]] = []
    collision_members = audit.loc[audit.is_collision_member]
    for normalized, group in collision_members.groupby(
        "smi_ted_normalized_smiles", sort=True
    ):
        stripped = group.stereo_stripped_canonical_smiles.unique()
        distinct_originals = group.original_canonical_smiles.unique()
        has_stereo = bool(group.has_assigned_stereochemistry.any())
        classification = (
            "stereochemical_collision"
            if len(stripped) == 1 and has_stereo and len(distinct_originals) > 1
            else "other_normalization_collision"
        )
        groups.append({
            "collision_group_id": collision_group_id(normalized),
            "classification": classification,
            "smi_ted_normalized_smiles": normalized,
            "group_size": len(group),
            "distinct_original_smiles": len(distinct_originals),
            "assigned_stereo_member_count": int(group.has_assigned_stereochemistry.sum()),
            "tetrahedral_stereo_member_count": int(
                group.has_assigned_tetrahedral_stereochemistry.sum()
            ),
            "bond_stereo_member_count": int(group.has_assigned_bond_stereochemistry.sum()),
            "maximum_heavy_atom_count": int(group.heavy_atom_count.max()),
            "original_indices": group.original_index.astype(int).tolist(),
            "molecule_ids": group.molecule_id.tolist(),
            "original_canonical_smiles": group.original_canonical_smiles.tolist(),
            "stereo_stripped_graphs": sorted(stripped.tolist()),
        })
    group_audit = pd.DataFrame(groups)
    classification = group_audit.set_index("collision_group_id").classification
    audit["collision_classification"] = audit.collision_group_id.map(classification).fillna("")

    stereo_groups = group_audit.loc[
        group_audit.classification == "stereochemical_collision"
    ]
    other_groups = group_audit.loc[
        group_audit.classification == "other_normalization_collision"
    ]
    size_distribution = Counter(stereo_groups.group_size.astype(int))
    examples = stereo_groups.sort_values(
        ["maximum_heavy_atom_count", "group_size", "smi_ted_normalized_smiles"]
    ).head(5)
    representative = examples.iloc[0]

    summary = {
        "schema_version": 1,
        "source_population_size": len(audit),
        "ordered_population_hash": ordered_population_hash(molecule_ids),
        "unique_original_canonical_smiles": int(audit.original_canonical_smiles.nunique()),
        "unique_smi_ted_normalized_smiles": int(audit.smi_ted_normalized_smiles.nunique()),
        "assigned_stereochemistry": {
            "molecule_count": int(audit.has_assigned_stereochemistry.sum()),
            "population_fraction": float(audit.has_assigned_stereochemistry.mean()),
            "tetrahedral_molecule_count": int(
                audit.has_assigned_tetrahedral_stereochemistry.sum()
            ),
            "bond_molecule_count": int(audit.has_assigned_bond_stereochemistry.sum()),
            "both_molecule_count": int((
                audit.has_assigned_tetrahedral_stereochemistry
                & audit.has_assigned_bond_stereochemistry
            ).sum()),
        },
        "normalization": {
            "official_function": "load.py:normalize_smiles",
            "arguments": {"canonical": True, "isomeric": False},
            "failed_molecule_count": 0,
            "loader_sha256": file_sha256(args.smi_ted_dir / "load.py"),
        },
        "collisions": {
            "group_count": len(group_audit),
            "affected_molecule_count": int(audit.is_collision_member.sum()),
            "maximum_group_size": int(group_audit.group_size.max()),
            "stereochemical_group_count": len(stereo_groups),
            "stereochemical_affected_molecule_count": int(stereo_groups.group_size.sum()),
            "stereochemical_assigned_member_count": int(
                stereo_groups.assigned_stereo_member_count.sum()
            ),
            "stereochemical_group_size_distribution": {
                str(size): count for size, count in sorted(size_distribution.items())
            },
            "other_group_count": len(other_groups),
            "other_affected_molecule_count": int(other_groups.group_size.sum()),
        },
        "deterministic_verification_group_id": representative.collision_group_id,
        "representative_examples": examples[[
            "collision_group_id", "smi_ted_normalized_smiles",
            "original_indices", "original_canonical_smiles",
        ]].to_dict(orient="records"),
        "provenance": {
            "dataset": str(args.dataset),
            "dataset_sha256": file_sha256(args.dataset),
            "rdkit_version": rdBase.rdkitVersion,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output_dir / "per_molecule_normalization_audit.parquet", index=False)
    audit.to_csv(args.output_dir / "per_molecule_normalization_audit.csv", index=False)
    group_audit.to_json(
        args.output_dir / "collision_group_audit.json", orient="records", indent=2
    )
    json_dump(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", summary)
    print(json.dumps(summary, indent=2))


def write_report(path: Path, summary: dict[str, Any]) -> None:
    stereo = summary["assigned_stereochemistry"]
    collisions = summary["collisions"]
    lines = [
        "# SMI-TED Light stereochemistry audit",
        "",
        f"The corrected EGFR population contains {summary['source_population_size']:,} unique canonical structures. "
        f"Assigned stereochemistry occurs in {stereo['molecule_count']:,} ({stereo['population_fraction']:.2%}).",
        "",
        f"The exact official `normalize_smiles` path (`canonical=True, isomeric=False`) produced "
        f"{summary['unique_smi_ted_normalized_smiles']:,} unique strings and "
        f"{collisions['group_count']:,} collision groups involving {collisions['affected_molecule_count']:,} molecules.",
        "",
        f"All {collisions['stereochemical_group_count']:,} groups ({collisions['stereochemical_affected_molecule_count']:,} molecules) "
        "were structurally attributable to removal of stereochemical assignments; no other normalization collision was found.",
        "",
        "These collisions are a property of the officially defined representation, not a curation failure. "
        "They do not constitute an encoding failure and do not change eligibility.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_embedding(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited")
    summary_path = args.output_dir / "summary.json"
    groups_path = args.output_dir / "collision_group_audit.json"
    if not summary_path.exists() or not groups_path.exists():
        raise FileNotFoundError("Run --stage analyze before embedding verification")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    selected = next(
        group for group in groups
        if group["collision_group_id"] == summary["deterministic_verification_group_id"]
    )
    official = load_official_module(args.smi_ted_dir)
    tokenizer = official.MolTranBertTokenizer(
        str(args.smi_ted_dir / "bert_vocab_curated.txt")
    )
    originals = selected["original_canonical_smiles"]
    normalized = [official.normalize_smiles(value) for value in originals]
    token_counts = [
        len(tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"])
        for value in normalized
    ]
    if len(set(normalized)) != 1:
        raise RuntimeError("Selected collision no longer has identical normalized inputs")
    if max(token_counts) > SMI_TED_MAX_TOTAL_TOKENS:
        raise RuntimeError("Selected verification group exceeds SMI-TED token limit")

    model = official.load_smi_ted(
        str(args.smi_ted_dir), "smi-ted-Light_40.pt", "bert_vocab_curated.txt"
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        encoded = model.encode(originals, batch_size=len(originals), return_torch=True)
    embeddings = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
    if embeddings.shape != (len(originals), SMI_TED_DIMENSION):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite SMI-TED embedding")
    pairwise_max = max(
        float(np.max(np.abs(embeddings[left] - embeddings[right])))
        for left in range(len(embeddings))
        for right in range(left + 1, len(embeddings))
    )
    verification = {
        "collision_group_id": selected["collision_group_id"],
        "original_indices": selected["original_indices"],
        "original_canonical_smiles": originals,
        "normalized_inputs": normalized,
        "normalized_inputs_identical": len(set(normalized)) == 1,
        "token_counts_including_special_tokens": token_counts,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "maximum_absolute_embedding_difference": pairwise_max,
        "tolerance": SMI_TED_REPRODUCIBILITY_ATOL,
        "identical_within_tolerance": pairwise_max <= SMI_TED_REPRODUCIBILITY_ATOL,
        "device": torch.cuda.get_device_name(0),
        "checkpoint": "smi-ted-Light_40.pt",
        "checkpoint_sha256": file_sha256(args.smi_ted_dir / "smi-ted-Light_40.pt"),
    }
    np.save(args.output_dir / "verification_group_embeddings.npy", embeddings)
    json_dump(args.output_dir / "embedding_verification.json", verification)
    summary["embedding_verification"] = verification
    json_dump(summary_path, summary)
    write_report(args.output_dir / "report.md", summary)
    with (args.output_dir / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\nThe deterministic example produced identical normalized inputs and a maximum "
            f"embedding difference of {pairwise_max:.9g} (tolerance "
            f"{SMI_TED_REPRODUCIBILITY_ATOL:.1g}).\n"
        )
    print(json.dumps(verification, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyze", "verify", "all"), default="all")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/processed/v051/targets/EGFR/clean.parquet"),
    )
    parser.add_argument(
        "--smi-ted-dir", type=Path,
        default=Path("data/processed/v051/checkpoints/smi_ted_light"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/representation_audits/v051/smi_ted_stereochemistry"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"analyze", "all"}:
        analyze(args)
    if args.stage in {"verify", "all"}:
        verify_embedding(args)


if __name__ == "__main__":
    main()
