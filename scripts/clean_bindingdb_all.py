import argparse
import re
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


def clean_chunk(chunk):
    chunk = chunk.dropna(
        subset=["Target Name", "Ligand SMILES"]
    ).copy()

    affinity_types = []
    affinity_values = []

    for _, row in chunk.iterrows():
        affinity_type, value = select_best_affinity(row)
        affinity_types.append(affinity_type)
        affinity_values.append(value)

    chunk["affinity_type"] = affinity_types
    chunk["aff_nM"] = affinity_values

    chunk = chunk.dropna(subset=["affinity_type", "aff_nM"]).copy()

    chunk["canonical_smiles"] = chunk["Ligand SMILES"].map(
        canonicalize_smiles
    )

    chunk = chunk.dropna(subset=["canonical_smiles"]).copy()

    return chunk[
        [
            "Target Name",
            "Ligand SMILES",
            "canonical_smiles",
            "affinity_type",
            "aff_nM",
        ]
    ]


def affinity_to_delta_g_kcal_mol(aff_nM, temperature=298.15):
    """
    Convert affinity in nM to approximate binding free energy.

    ΔG = RT ln(K)
    K must be in molar.
    """
    R = 0.0019872041  # kcal mol-1 K-1
    aff_M = aff_nM * 1e-9

    return R * temperature * np.log(aff_M)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip-path",
        default="data/raw/BindingDB_All_202607_tsv.zip",
    )

    parser.add_argument(
        "--output-path",
        default="data/processed/bindingdb_clean_all.parquet",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = [
        "Target Name",
        "Ligand SMILES",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
    ]

    cleaned_chunks = []

    with zipfile.ZipFile(args.zip_path) as zf:
        tsv_name = zf.namelist()[0]

        with zf.open(tsv_name) as f:
            reader = pd.read_csv(
                f,
                sep="\t",
                usecols=usecols,
                chunksize=args.chunksize,
                low_memory=False,
            )

            for i, chunk in enumerate(reader):
                cleaned = clean_chunk(chunk)

                cleaned_chunks.append(cleaned)

                print(
                    f"Chunk {i:04d}: "
                    f"raw={len(chunk):,}, cleaned={len(cleaned):,}"
                )

    df = pd.concat(cleaned_chunks, ignore_index=True)

    grouped = (
        df
        .groupby(
            ["Target Name", "canonical_smiles", "affinity_type"],
            as_index=False,
        )
        .agg(
            aff_nM_median=("aff_nM", "median"),
            n_meas=("aff_nM", "size"),
            **{"Ligand SMILES": ("Ligand SMILES", "first")},
        )
    )

    grouped["DG_median"] = affinity_to_delta_g_kcal_mol(
        grouped["aff_nM_median"]
    )

    grouped["label"] = (
        grouped["aff_nM_median"] <= 1000
    ).astype(int)

    grouped = grouped[
        [
            "Target Name",
            "Ligand SMILES",
            "canonical_smiles",
            "affinity_type",
            "aff_nM_median",
            "DG_median",
            "n_meas",
            "label",
        ]
    ]

    grouped.to_parquet(output_path, index=False)

    print("\nSaved:", output_path)
    print("Shape:", grouped.shape)
    print("Active fraction:", grouped["label"].mean())
    print("Targets:", grouped["Target Name"].nunique())


if __name__ == "__main__":
    main()