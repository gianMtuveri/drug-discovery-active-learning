#!/usr/bin/env python3
"""Compare immutable v0.5.0 EGFR artifacts with corrected v0.5.1 curation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIORITY = {"Ki": 0, "Kd": 1, "IC50": 2}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def affinity_summary(values: pd.Series) -> dict[str, float | int]:
    values = values.astype(float)
    return {
        "count": int(len(values)),
        "mean_nM": float(values.mean()),
        "std_nM": float(values.std(ddof=1)),
        "min_nM": float(values.min()),
        "p25_nM": float(values.quantile(0.25)),
        "median_nM": float(values.median()),
        "p75_nM": float(values.quantile(0.75)),
        "max_nM": float(values.max()),
    }


def historical_priority_reference(original: pd.DataFrame) -> pd.DataFrame:
    """Choose the established priority from historical per-type aggregates."""
    reference = original.copy()
    reference["_priority"] = reference["affinity_type"].map(PRIORITY)
    if reference["_priority"].isna().any():
        raise ValueError("Historical EGFR data contains unsupported affinity types")
    return (
        reference.sort_values(["canonical_smiles", "_priority"], kind="stable")
        .drop_duplicates("canonical_smiles", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def audit(original_path: Path, corrected_path: Path, output_dir: Path) -> dict:
    original = pd.read_parquet(original_path)
    corrected = pd.read_parquet(corrected_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if corrected.canonical_smiles.duplicated().any():
        raise AssertionError("Corrected EGFR artifact contains duplicate structures")
    reference = historical_priority_reference(original)
    original_smiles = set(original.canonical_smiles)
    corrected_smiles = set(corrected.canonical_smiles)
    added = sorted(corrected_smiles - original_smiles)
    removed = sorted(original_smiles - corrected_smiles)

    comparison = reference[
        ["canonical_smiles", "affinity_type", "aff_nM_median"]
    ].rename(
        columns={
            "affinity_type": "historical_priority_type",
            "aff_nM_median": "historical_priority_affinity_nM",
        }
    ).merge(
        corrected[
            [
                "canonical_smiles",
                "affinity_type",
                "aff_nM_median",
                "available_affinity_types",
                "n_available_affinity_types",
            ]
        ].rename(
            columns={
                "affinity_type": "corrected_type",
                "aff_nM_median": "corrected_affinity_nM",
            }
        ),
        on="canonical_smiles",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    comparison["selected_type_agrees"] = (
        comparison.historical_priority_type == comparison.corrected_type
    )
    comparison["selected_affinity_agrees"] = np.isclose(
        comparison.historical_priority_affinity_nM,
        comparison.corrected_affinity_nM,
        rtol=1e-12,
        atol=0.0,
        equal_nan=False,
    )
    comparison["crosses_500nM_vs_historical_priority"] = (
        (comparison.historical_priority_affinity_nM <= 500)
        != (comparison.corrected_affinity_nM <= 500)
    )
    comparison.to_csv(output_dir / "molecule_level_comparison.csv", index=False)
    pd.DataFrame({"canonical_smiles": added}).to_csv(
        output_dir / "unexpected_added_structures.csv", index=False
    )
    pd.DataFrame({"canonical_smiles": removed}).to_csv(
        output_dir / "unexpected_removed_structures.csv", index=False
    )

    old_groups = original.groupby("canonical_smiles", sort=False)
    old_group_sizes = old_groups.size()
    duplicate_groups = old_group_sizes[old_group_sizes > 1].index
    duplicated = original[original.canonical_smiles.isin(duplicate_groups)]
    duplicated_groups = duplicated.groupby("canonical_smiles", sort=False)
    historical_threshold_ambiguity = int(
        (
            (duplicated_groups.aff_nM_median.min() <= 500)
            & (duplicated_groups.aff_nM_median.max() > 500)
        ).sum()
    )

    expected_types = corrected.available_affinity_types.str.split("|").str[0]
    priority_agreement = corrected.affinity_type.eq(expected_types)
    manifest = {
        "historical_reference_definition": (
            "For each canonical structure, choose its existing v0.5.0 per-type "
            "aggregate using Ki > Kd > IC50; do not choose an arbitrary row."
        ),
        "original_artifact": str(original_path),
        "original_artifact_sha256": file_sha256(original_path),
        "corrected_artifact": str(corrected_path),
        "corrected_artifact_sha256": file_sha256(corrected_path),
        "original_record_count": int(len(original)),
        "corrected_record_count": int(len(corrected)),
        "original_unique_canonical_structures": int(original.canonical_smiles.nunique()),
        "corrected_unique_canonical_structures": int(corrected.canonical_smiles.nunique()),
        "original_excess_duplicate_rows": int(original.canonical_smiles.duplicated().sum()),
        "original_duplicate_structure_groups": int(len(duplicate_groups)),
        "corrected_excess_duplicate_rows": int(corrected.canonical_smiles.duplicated().sum()),
        "measurement_type_counts_original_rows": {
            str(key): int(value) for key, value in original.affinity_type.value_counts().items()
        },
        "measurement_type_counts_historical_priority_reference": {
            str(key): int(value) for key, value in reference.affinity_type.value_counts().items()
        },
        "measurement_type_counts_corrected": {
            str(key): int(value) for key, value in corrected.affinity_type.value_counts().items()
        },
        "selected_type_differences_vs_historical_priority": int(
            (~comparison.selected_type_agrees).sum()
        ),
        "selected_affinity_differences_vs_historical_priority": int(
            (~comparison.selected_affinity_agrees).sum()
        ),
        "threshold_crossings_vs_historical_priority": int(
            comparison.crosses_500nM_vs_historical_priority.sum()
        ),
        "historical_duplicate_groups_spanning_500nM": historical_threshold_ambiguity,
        "unexpected_added_structures": int(len(added)),
        "unexpected_removed_structures": int(len(removed)),
        "priority_rule_agreement_count": int(priority_agreement.sum()),
        "priority_rule_disagreement_count": int((~priority_agreement).sum()),
        "affinity_distributions": {
            "original_row_weighted": affinity_summary(original.aff_nM_median),
            "historical_priority_one_per_structure": affinity_summary(reference.aff_nM_median),
            "corrected_one_per_structure": affinity_summary(corrected.aff_nM_median),
        },
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# EGFR v0.5.0 versus corrected v0.5.1 curation",
        "",
        manifest["historical_reference_definition"],
        "",
        f"- Records: {len(original):,} original; {len(corrected):,} corrected.",
        f"- Unique structures: {original.canonical_smiles.nunique():,} original; "
        f"{corrected.canonical_smiles.nunique():,} corrected.",
        f"- Duplicate excess rows: {manifest['original_excess_duplicate_rows']} original; "
        f"{manifest['corrected_excess_duplicate_rows']} corrected.",
        f"- Selected affinity differences from the historical priority reference: "
        f"{manifest['selected_affinity_differences_vs_historical_priority']}.",
        f"- 500 nM crossings from that reference: "
        f"{manifest['threshold_crossings_vs_historical_priority']}.",
        f"- Historical duplicate groups spanning 500 nM: "
        f"{historical_threshold_ambiguity}.",
        f"- Unexpected structures added/removed: {len(added)}/{len(removed)}.",
        f"- Priority-rule disagreements: {manifest['priority_rule_disagreement_count']}.",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/processed/targets/EGFR/clean.parquet"),
    )
    parser.add_argument(
        "--corrected",
        type=Path,
        default=Path("data/processed/v051/targets/EGFR/clean.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/representation_audit/v051_egfr_curation"),
    )
    args = parser.parse_args()
    result = audit(args.original, args.corrected, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
