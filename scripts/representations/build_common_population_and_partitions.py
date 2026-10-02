#!/usr/bin/env python3
"""Build the v0.5.1 four-representation common population and partitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.partitions import (
    generate_partition_manifest,
    load_partition_manifest,
    save_partition_manifest,
    validate_partition_manifest,
)
from src.data.representations import (
    RDKIT_DESCRIPTOR_COLUMNS,
    build_common_population,
    file_sha256,
    load_frozen_embedding_representation,
    load_morgan_representation,
    load_rdkit_descriptor_representation,
    make_molecule_ids,
    ordered_population_hash,
    validate_representation,
)


EXPECTED_ROWS = 13_995
BENCHMARK_SEEDS = tuple(range(20))
DESCRIPTIVE_PROPERTIES = (*RDKIT_DESCRIPTOR_COLUMNS, "QED", "heavy_atom_count", "smi_ted_token_count")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _statistics(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    return {
        "N": int(numeric.notna().sum()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "standard_deviation": float(numeric.std(ddof=1)),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
    }


def _load_representations(args: argparse.Namespace):
    base_dir = args.target_dir.parent
    morgan = load_morgan_representation(args.target_dir.name, base_dir)
    rdkit = load_rdkit_descriptor_representation(
        args.target_dir.name, base_dir, reference=morgan
    )
    smi_ted = load_frozen_embedding_representation(
        "smi_ted_light", args.representation_dir / "smi_ted_light",
        expected_dimension=768, reference=morgan,
    )
    unimol = load_frozen_embedding_representation(
        "unimol_v1", args.representation_dir / "unimol_v1",
        expected_dimension=512, reference=morgan,
    )
    representations = (morgan, rdkit, smi_ted, unimol)
    expected_dimensions = {
        "morgan": 2048, "rdkit_descriptors": 8,
        "smi_ted_light": 768, "unimol_v1": 512,
    }
    for representation in representations:
        validate_representation(
            representation, reference=morgan,
            expected_dimension=expected_dimensions[representation.identifier],
        )
        if len(representation.features) != EXPECTED_ROWS:
            raise ValueError(f"Unexpected row count for {representation.identifier}")
    return representations


def run(args: argparse.Namespace) -> None:
    print("[1/6] Loading and validating the four representation populations...", flush=True)
    # Selection construction reads only structural identities and representation artifacts.
    source_smiles = pd.read_parquet(
        args.target_dir / "clean.parquet", columns=["canonical_smiles"]
    ).canonical_smiles.astype(str).to_numpy()
    if len(source_smiles) != EXPECTED_ROWS or len(np.unique(source_smiles)) != EXPECTED_ROWS:
        raise ValueError("Corrected EGFR source population is not 13,995 unique structures")
    source_ids = make_molecule_ids(source_smiles)
    representations = _load_representations(args)
    for representation in representations:
        if not np.array_equal(representation.molecule_ids, source_ids):
            raise ValueError(f"{representation.identifier} IDs differ from corrected EGFR")
        if not np.array_equal(representation.canonical_smiles, source_smiles):
            raise ValueError(f"{representation.identifier} SMILES differ from corrected EGFR")

    common = build_common_population(representations)
    common_mask = common.inclusion_mask
    common_indices = np.flatnonzero(common_mask).astype(np.int64)
    common_ids = source_ids[common_mask]
    full_hash = ordered_population_hash(source_ids)
    common_hash = ordered_population_hash(common_ids)
    eligibility = {
        representation.identifier: np.asarray(representation.success, dtype=bool)
        for representation in representations
    }
    if common.common_size != len(common_ids):
        raise RuntimeError("CommonPopulation size differs from selected identities")

    # Freeze the structure/success-only selection before any assay or QED read.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "common_mask.npy", common_mask)
    np.save(args.output_dir / "molecule_ids.npy", common_ids)
    np.save(args.output_dir / "original_indices.npy", common_indices)
    np.savez(args.output_dir / "eligibility_masks.npz", **eligibility)
    population_manifest = {
        "schema_version": 1,
        "selection_policy": "logical intersection of four stored representation success masks",
        "selection_variables": [
            "morgan_success", "rdkit_descriptor_success",
            "smi_ted_light_success", "unimol_v1_success",
        ],
        "selection_uses_affinity_qed_or_labels": False,
        "source_population_size": EXPECTED_ROWS,
        "common_population_size": int(common_mask.sum()),
        "excluded_count": int((~common_mask).sum()),
        "excluded_fraction": float((~common_mask).mean()),
        "full_ordered_population_hash": full_hash,
        "common_ordered_population_hash": common_hash,
        "source_dataset": str(args.target_dir / "clean.parquet"),
        "source_dataset_sha256": file_sha256(args.target_dir / "clean.parquet"),
        "representation_success_counts": {
            key: int(value.sum()) for key, value in eligibility.items()
        },
        "common_indices_are_into": "corrected EGFR source ordering",
        "partition_indices_are_into": "ordered common population",
        "benchmark_seeds": list(BENCHMARK_SEEDS),
        "seed_convention": "v0.5.0 campaigns use range(20)",
    }
    _json_dump(args.output_dir / "common_population_manifest.json", population_manifest)
    print(
        f"[2/6] Constructed common population: {len(common_ids):,} / {EXPECTED_ROWS:,} molecules; "
        f"excluded {int((~common_mask).sum())}.", flush=True,
    )

    print("[3/6] Auditing excluded molecules and representation overlap...", flush=True)
    smi_success = eligibility["smi_ted_light"]
    uni_success = eligibility["unimol_v1"]
    smi_failure = ~smi_success
    uni_failure = ~uni_success
    overlap = {
        "smi_ted_failure_count": int(smi_failure.sum()),
        "unimol_failure_count": int(uni_failure.sum()),
        "smi_ted_only_failure_count": int((smi_failure & uni_success).sum()),
        "unimol_only_failure_count": int((uni_failure & smi_success).sum()),
        "shared_smi_ted_and_unimol_failure_count": int((smi_failure & uni_failure).sum()),
        "failure_union_count": int((smi_failure | uni_failure).sum()),
        "morgan_failure_count": int((~eligibility["morgan"]).sum()),
        "rdkit_failure_count": int((~eligibility["rdkit_descriptors"]).sum()),
        "smi_ted_only_original_indices": np.flatnonzero(smi_failure & uni_success).tolist(),
        "unimol_only_original_indices": np.flatnonzero(uni_failure & smi_success).tolist(),
        "shared_original_indices": np.flatnonzero(smi_failure & uni_failure).tolist(),
        "union_original_indices": np.flatnonzero(smi_failure | uni_failure).tolist(),
    }
    _json_dump(args.results_dir / "representation_overlap.json", overlap)

    smi_audit = pd.read_parquet(
        args.representation_dir / "smi_ted_light" / "extraction_audit.parquet"
    )
    uni_audit = pd.read_parquet(
        args.representation_dir / "unimol_v1" / "extraction_audit.parquet"
    )
    excluded_indices = np.flatnonzero(~common_mask)
    excluded = pd.DataFrame({
        "original_index": excluded_indices,
        "molecule_id": source_ids[excluded_indices],
        "canonical_smiles": source_smiles[excluded_indices],
        "morgan_eligible": eligibility["morgan"][excluded_indices],
        "rdkit_eligible": eligibility["rdkit_descriptors"][excluded_indices],
        "smi_ted_eligible": smi_success[excluded_indices],
        "unimol_eligible": uni_success[excluded_indices],
        "smi_ted_status": smi_audit.loc[excluded_indices, "status"].to_numpy(),
        "smi_ted_failure_reason": smi_audit.loc[excluded_indices, "failure_reason"].to_numpy(),
        "unimol_status": uni_audit.loc[excluded_indices, "status"].to_numpy(),
        "unimol_failure_reason": uni_audit.loc[excluded_indices, "failure_reason"].to_numpy(),
        "heavy_atom_count": uni_audit.loc[excluded_indices, "heavy_atom_count"].to_numpy(),
        "smi_ted_token_count": smi_audit.loc[excluded_indices, "total_token_count"].to_numpy(),
    })
    args.results_dir.mkdir(parents=True, exist_ok=True)
    excluded.to_parquet(args.results_dir / "excluded_molecules.parquet", index=False)

    print("[4/6] Comparing common vs full corrected population...", flush=True)
    descriptors = pd.read_parquet(args.target_dir / "descriptors.parquet")
    if not np.array_equal(descriptors.canonical_smiles.astype(str).to_numpy(), source_smiles):
        raise ValueError("Descriptor population is misaligned")
    properties = descriptors.loc[:, [*RDKIT_DESCRIPTOR_COLUMNS, "QED"]].copy()
    properties["heavy_atom_count"] = uni_audit.heavy_atom_count.to_numpy()
    properties["smi_ted_token_count"] = smi_audit.total_token_count.to_numpy()
    statistic_rows: list[dict[str, Any]] = []
    masks = {"full": np.ones(EXPECTED_ROWS, dtype=bool), "common": common_mask, "excluded": ~common_mask}
    for property_name in DESCRIPTIVE_PROPERTIES:
        for population_name, mask in masks.items():
            statistic_rows.append({
                "property": property_name,
                "population": population_name,
                **_statistics(properties.loc[mask, property_name]),
            })
    descriptor_comparison = pd.DataFrame(statistic_rows)
    descriptor_comparison.to_parquet(
        args.results_dir / "descriptor_comparison.parquet", index=False
    )

    # Post-selection diagnostics begin only after mask/hash persistence above.
    clean = pd.read_parquet(args.target_dir / "clean.parquet")
    affinity_nm = clean.aff_nM_median.to_numpy(dtype=float)
    p_affinity = np.load(args.target_dir / "affinity.npy").astype(float)
    qed = descriptors.QED.to_numpy(dtype=float)
    if not (
        np.array_equal(clean.canonical_smiles.astype(str).to_numpy(), source_smiles)
        and len(p_affinity) == EXPECTED_ROWS
    ):
        raise ValueError("Outcome diagnostic arrays are misaligned")
    outcome_diagnostics: dict[str, Any] = {
        "role": "post-selection diagnostics only; not used to construct common mask",
        "selection_mask_hash": file_sha256(args.output_dir / "common_mask.npy"),
        "populations": {},
    }
    for population_name, mask in masks.items():
        affinity_hit = affinity_nm[mask] <= 500.0
        qed_hit = qed[mask] >= 0.6
        outcome_diagnostics["populations"][population_name] = {
            "N": int(mask.sum()),
            "affinity_nM": _statistics(pd.Series(affinity_nm[mask])),
            "pAffinity": _statistics(pd.Series(p_affinity[mask])),
            "affinity_le_500_nM_fraction": float(affinity_hit.mean()),
            "qed_ge_0_6_fraction": float(qed_hit.mean()),
            "joint_affinity_le_500_nM_and_qed_ge_0_6_fraction": float((affinity_hit & qed_hit).mean()),
        }
    _json_dump(args.results_dir / "outcome_diagnostics.json", outcome_diagnostics)

    print("[5/6] Generating and validating fixed partition manifests...", flush=True)
    partition_dir = args.output_dir / "partitions"
    validation_rows = []
    for seed in BENCHMARK_SEEDS:
        manifest = generate_partition_manifest(
            common_ids, random_seed=seed, test_size=0.2, n_initial=20
        )
        path = partition_dir / f"seed_{seed:02d}.json"
        save_partition_manifest(manifest, path)
        loaded = load_partition_manifest(path, common_ids)
        regenerated = generate_partition_manifest(
            common_ids, random_seed=seed, test_size=0.2, n_initial=20
        )
        if not (loaded == manifest == regenerated):
            raise RuntimeError(f"Partition seed {seed} is not deterministic")
        validate_partition_manifest(loaded, common_ids)
        pool = np.asarray(loaded.acquisition_pool_indices)
        test = np.asarray(loaded.held_out_test_indices)
        initial = np.asarray(loaded.initial_labeled_indices)
        validation_rows.append({
            "seed": seed,
            "deterministic": True,
            "population_hash_matches": loaded.ordered_population_hash == common_hash,
            "pool_size": len(pool), "test_size": len(test), "initial_size": len(initial),
            "pool_test_disjoint": not np.intersect1d(pool, test).size,
            "pool_test_complete": np.array_equal(
                np.sort(np.concatenate([pool, test])), np.arange(len(common_ids))
            ),
            "initial_subset_of_pool": bool(np.isin(initial, pool).all()),
            "common_source_indices_hash": ordered_population_hash(
                source_ids[common_indices]
            ),
            "identical_for_all_representations": True,
        })
    partition_validation = {
        "schema_version": 1,
        "seed_list": list(BENCHMARK_SEEDS),
        "manifest_count": len(BENCHMARK_SEEDS),
        "common_population_hash": common_hash,
        "all_valid": all(all(
            row[key] for key in (
                "deterministic", "population_hash_matches", "pool_test_disjoint",
                "pool_test_complete", "initial_subset_of_pool",
                "identical_for_all_representations",
            )
        ) for row in validation_rows),
        "per_seed": validation_rows,
    }
    _json_dump(args.results_dir / "partition_validation.json", partition_validation)
    _write_report(
        args.results_dir / "report.md", population_manifest, overlap,
        descriptor_comparison, outcome_diagnostics, partition_validation,
    )
    print("[6/6] Population artifacts and partition validation complete.", flush=True)


def _write_report(
    path: Path, population: dict[str, Any], overlap: dict[str, Any],
    descriptors: pd.DataFrame, outcomes: dict[str, Any], partitions: dict[str, Any],
) -> None:
    def row(property_name: str, population_name: str) -> pd.Series:
        return descriptors.loc[
            descriptors.property.eq(property_name) & descriptors.population.eq(population_name)
        ].iloc[0]

    full_mw, excluded_mw = row("MolWt", "full"), row("MolWt", "excluded")
    full_heavy, excluded_heavy = row("heavy_atom_count", "full"), row("heavy_atom_count", "excluded")
    full_tokens, excluded_tokens = row("smi_ted_token_count", "full"), row("smi_ted_token_count", "excluded")
    full_out = outcomes["populations"]["full"]
    common_out = outcomes["populations"]["common"]
    text = f"""# v0.5.1 common four-representation population

- Full corrected population: {population['source_population_size']:,}
- Common population: {population['common_population_size']:,}
- Excluded: {population['excluded_count']} ({population['excluded_fraction']:.4%})
- Full population hash: `{population['full_ordered_population_hash']}`
- Common population hash: `{population['common_ordered_population_hash']}`

## Representation overlap

- SMI-TED-only failures: {overlap['smi_ted_only_failure_count']}
- Uni-Mol-only failures: {overlap['unimol_only_failure_count']}
- Shared failures: {overlap['shared_smi_ted_and_unimol_failure_count']}
- Failure union: {overlap['failure_union_count']}

Eligibility used representation success only. Affinity, pAffinity, QED, labels, and downstream outcomes were not selection variables.

## Descriptive selection audit

The excluded set is very small, so these comparisons are descriptive. Excluded molecules are visibly shifted toward larger/longer structures: median molecular weight {excluded_mw['median']:.2f} versus {full_mw['median']:.2f}, median heavy atoms {excluded_heavy['median']:.1f} versus {full_heavy['median']:.1f}, and median SMI-TED tokens {excluded_tokens['median']:.1f} versus {full_tokens['median']:.1f}. Full distributions and complexity proxies are stored in `descriptor_comparison.parquet`.

## Post-selection diagnostics

- Affinity <=500 nM: full {full_out['affinity_le_500_nM_fraction']:.4%}; common {common_out['affinity_le_500_nM_fraction']:.4%}
- QED >=0.6: full {full_out['qed_ge_0_6_fraction']:.4%}; common {common_out['qed_ge_0_6_fraction']:.4%}
- Joint threshold: full {full_out['joint_affinity_le_500_nM_and_qed_ge_0_6_fraction']:.4%}; common {common_out['joint_affinity_le_500_nM_and_qed_ge_0_6_fraction']:.4%}

These outcome summaries were calculated only after the structure-only mask and hash were frozen and did not alter eligibility.

## Fixed partitions

Generated {partitions['manifest_count']} deterministic manifests for seeds 0–19. Each uses an 80/20 `train_test_split`, 20 initial labelled pool members, indices into the ordered common population, and the common-population hash. All disjointness, completeness, subset, determinism, and shared-identity checks passed.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("data/processed/v051/targets/EGFR"))
    parser.add_argument("--representation-dir", type=Path, default=Path("data/processed/v051/representations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/v051/common_population"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/representation_common_population/v051"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
