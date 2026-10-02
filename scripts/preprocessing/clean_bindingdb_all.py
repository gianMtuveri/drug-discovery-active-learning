import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


AFFINITY_PRIORITY = [
    ("Ki", "Ki (nM)"),
    ("Kd", "Kd (nM)"),
    ("IC50", "IC50 (nM)"),
]

CURATION_POLICY_VERSION = "v051_molecule_target_priority_1"
HISTORICAL_OUTPUT_PATH = Path("data/processed/bindingdb_clean_all.parquet")


def parse_affinity_nM(value):
    """
    Parse BindingDB affinity values.

    Keeps simple numeric values and simple inequalities by extracting
    the numeric part.

    Examples
    --------
    '10'      -> 10.0
    '<10'     -> 10.0
    '>10000'  -> 10000.0
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    text = text.replace(",", "")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

    if match is None:
        return np.nan

    return float(match.group(0))


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def select_best_affinity(row):
    """
    Select one affinity measurement from a BindingDB row using priority:
    Ki > Kd > IC50.
    """

    for affinity_type, column in AFFINITY_PRIORITY:
        value = parse_affinity_nM(row[column])

        if not np.isnan(value) and value > 0:
            return affinity_type, value

    return None, np.nan


def extract_long_measurements(chunk):
    """Return every valid Ki, Kd, and IC50 measurement in long format.

    The parser and positivity rule are unchanged from ``select_best_affinity``.
    Unlike the historical implementation, lower-priority measurements are not
    discarded merely because the same source row also contains a Ki or Kd.
    """
    required = {
        "Target Name",
        "Ligand SMILES",
        *(column for _, column in AFFINITY_PRIORITY),
    }
    missing = sorted(required.difference(chunk.columns))
    if missing:
        raise ValueError(f"BindingDB input is missing required columns: {missing}")

    statistics = {
        "source_rows": int(len(chunk)),
        "missing_target_or_smiles_rows": 0,
        "rows_without_valid_measurement": 0,
        "invalid_structure_rows": 0,
        "valid_measurements": 0,
        "invalid_or_missing_measurement_cells": 0,
    }
    records = []
    for _, row in chunk.iterrows():
        if pd.isna(row["Target Name"]) or pd.isna(row["Ligand SMILES"]):
            statistics["missing_target_or_smiles_rows"] += 1
            continue

        measurements = []
        for affinity_type, column in AFFINITY_PRIORITY:
            value = parse_affinity_nM(row[column])
            if np.isfinite(value) and value > 0:
                measurements.append((affinity_type, value))
            else:
                statistics["invalid_or_missing_measurement_cells"] += 1

        if not measurements:
            statistics["rows_without_valid_measurement"] += 1
            continue

        canonical_smiles = canonicalize_smiles(row["Ligand SMILES"])
        if canonical_smiles is None:
            statistics["invalid_structure_rows"] += 1
            continue

        for affinity_type, value in measurements:
            records.append(
                {
                    "Target Name": row["Target Name"],
                    "Ligand SMILES": row["Ligand SMILES"],
                    "canonical_smiles": canonical_smiles,
                    "affinity_type": affinity_type,
                    "aff_nM": value,
                }
            )
        statistics["valid_measurements"] += len(measurements)

    return pd.DataFrame.from_records(
        records,
        columns=[
            "Target Name",
            "Ligand SMILES",
            "canonical_smiles",
            "affinity_type",
            "aff_nM",
        ],
    ), statistics


def aggregate_and_select_measurements(long_measurements):
    """Aggregate within type, then apply Ki > Kd > IC50 per molecule-target."""
    required = {
        "Target Name",
        "Ligand SMILES",
        "canonical_smiles",
        "affinity_type",
        "aff_nM",
    }
    missing = sorted(required.difference(long_measurements.columns))
    if missing:
        raise ValueError(f"Long-format measurements are missing columns: {missing}")
    if long_measurements.empty:
        raise ValueError("No valid BindingDB measurements were available for curation")

    per_type = (
        long_measurements.groupby(
            ["Target Name", "canonical_smiles", "affinity_type"],
            as_index=False,
        )
        .agg(
            aff_nM_median=("aff_nM", "median"),
            n_meas=("aff_nM", "size"),
            **{"Ligand SMILES": ("Ligand SMILES", "first")},
        )
    )
    priority = {name: position for position, (name, _) in enumerate(AFFINITY_PRIORITY)}
    per_type["_affinity_priority"] = per_type["affinity_type"].map(priority)
    if per_type["_affinity_priority"].isna().any():
        raise ValueError("Unsupported affinity type found after aggregation")

    availability = (
        per_type.sort_values("_affinity_priority")
        .groupby(["Target Name", "canonical_smiles"], as_index=False)
        .agg(
            available_affinity_types=(
                "affinity_type",
                lambda values: "|".join(values),
            ),
            n_available_affinity_types=("affinity_type", "size"),
        )
    )
    selected = (
        per_type.sort_values(
            ["Target Name", "canonical_smiles", "_affinity_priority"],
            kind="stable",
        )
        .drop_duplicates(["Target Name", "canonical_smiles"], keep="first")
        .merge(
            availability,
            on=["Target Name", "canonical_smiles"],
            how="left",
            validate="one_to_one",
        )
    )
    selected["cross_type_resolved"] = selected["n_available_affinity_types"] > 1
    selected["DG_median"] = affinity_to_delta_g_kcal_mol(selected["aff_nM_median"])
    selected["label"] = (selected["aff_nM_median"] <= 1000).astype(int)
    selected = selected[
        [
            "Target Name",
            "Ligand SMILES",
            "canonical_smiles",
            "affinity_type",
            "aff_nM_median",
            "DG_median",
            "n_meas",
            "label",
            "available_affinity_types",
            "n_available_affinity_types",
            "cross_type_resolved",
        ]
    ].sort_values(["Target Name", "canonical_smiles"], kind="stable", ignore_index=True)

    if selected.duplicated(["Target Name", "canonical_smiles"]).any():
        raise AssertionError("Corrected curation left duplicate molecule-target pairs")
    return selected, per_type.drop(columns="_affinity_priority")


def clean_chunk(chunk):
    """Compatibility wrapper returning the corrected long-format records."""
    return extract_long_measurements(chunk)[0]


def affinity_to_delta_g_kcal_mol(aff_nM, temperature=298.15):
    """
    Convert affinity in nM to approximate binding free energy.

    ΔG = RT ln(K)
    K must be in molar.
    """
    R = 0.0019872041  # kcal mol-1 K-1
    aff_M = aff_nM * 1e-9

    return R * temperature * np.log(aff_M)


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assert_safe_output(output_path, overwrite=False):
    output_path = Path(output_path)
    if output_path.resolve() == HISTORICAL_OUTPUT_PATH.resolve():
        raise ValueError(
            "Refusing to overwrite the historical v0.5.0 master artifact; "
            "use a path under data/processed/v051/"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite")
    return output_path


def curate_bindingdb_archive(
    zip_path,
    output_path,
    *,
    chunksize=100_000,
    overwrite=False,
):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing BindingDB source archive: {zip_path}")
    output_path = _assert_safe_output(output_path, overwrite=overwrite)
    audit_path = output_path.with_name(f"{output_path.stem}_measurement_audit.parquet")
    manifest_path = output_path.with_name("curation_manifest.json")
    for companion in (audit_path, manifest_path):
        if companion.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {companion}; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = [
        "Target Name",
        "Ligand SMILES",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
    ]
    cleaned_chunks = []
    exclusion_counts = {
        "source_rows": 0,
        "missing_target_or_smiles_rows": 0,
        "rows_without_valid_measurement": 0,
        "invalid_structure_rows": 0,
        "valid_measurements": 0,
        "invalid_or_missing_measurement_cells": 0,
    }
    with zipfile.ZipFile(zip_path) as zf:
        tsv_names = [name for name in zf.namelist() if not name.endswith("/")]
        if len(tsv_names) != 1:
            raise ValueError("Expected exactly one source table in BindingDB archive")
        with zf.open(tsv_names[0]) as handle:
            reader = pd.read_csv(
                handle,
                sep="\t",
                usecols=usecols,
                chunksize=chunksize,
                low_memory=False,
            )
            for index, chunk in enumerate(reader):
                cleaned, statistics = extract_long_measurements(chunk)
                cleaned_chunks.append(cleaned)
                for key, value in statistics.items():
                    exclusion_counts[key] += value
                print(
                    f"Chunk {index:04d}: raw={len(chunk):,}, "
                    f"valid_measurements={len(cleaned):,}"
                )

    long_measurements = pd.concat(cleaned_chunks, ignore_index=True)
    selected, per_type = aggregate_and_select_measurements(long_measurements)
    selected.to_parquet(output_path, index=False)

    per_type.to_parquet(audit_path, index=False)

    selected_counts = {
        affinity_type: int((selected["affinity_type"] == affinity_type).sum())
        for affinity_type, _ in AFFINITY_PRIORITY
    }
    manifest = {
        "curation_policy_version": CURATION_POLICY_VERSION,
        "measurement_priority": [name for name, _ in AFFINITY_PRIORITY],
        "aggregation_method": "median within Target Name + canonical_smiles + affinity_type",
        "selection_unit": "Target Name + canonical_smiles",
        "source_archive_path": str(zip_path),
        "source_archive_sha256": _file_sha256(zip_path),
        "git_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "measurement_audit_path": str(audit_path),
        "input_source_rows": exclusion_counts["source_rows"],
        "valid_long_measurements": int(len(long_measurements)),
        "per_type_aggregate_rows": int(len(per_type)),
        "output_rows": int(len(selected)),
        "unique_molecule_target_pairs": int(
            selected[["Target Name", "canonical_smiles"]].drop_duplicates().shape[0]
        ),
        "cross_type_groups_resolved": int(selected["cross_type_resolved"].sum()),
        "selected_affinity_type_counts": selected_counts,
        "exclusion_counts": {
            key: int(value)
            for key, value in exclusion_counts.items()
            if key not in {"source_rows", "valid_measurements"}
        },
        "parser_policy": {
            "units": "nM source columns only",
            "positive_values_only": True,
            "simple_inequalities": "numeric bound retained, matching historical parser",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return selected, manifest


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip-path",
        default="data/raw/BindingDB_All_202607_tsv.zip",
    )

    parser.add_argument(
        "--output-path",
        default="data/processed/v051/bindingdb_clean_all.parquet",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    selected, manifest = curate_bindingdb_archive(
        args.zip_path,
        args.output_path,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    print("\nSaved:", args.output_path)
    print("Shape:", selected.shape)
    print("Active fraction:", selected["label"].mean())
    print("Targets:", selected["Target Name"].nunique())
    print("Cross-type groups resolved:", manifest["cross_type_groups_resolved"])


if __name__ == "__main__":
    main()
