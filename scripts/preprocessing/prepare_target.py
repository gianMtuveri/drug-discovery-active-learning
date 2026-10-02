import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.chemistry.fingerprints import featurize_smiles
from src.chemistry.descriptors import compute_basic_descriptors


HISTORICAL_TARGETS_DIR = Path("data/processed/targets")


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_target_registry(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_target(df, query, match="contains"):
    target_names = df["Target Name"].astype(str)

    if match == "exact":
        mask = target_names.str.lower() == query.lower()

    elif match == "contains":
        mask = target_names.str.contains(
            query,
            case=False,
            na=False,
            regex=False,
        )

    else:
        raise ValueError("match must be one of: 'exact', 'contains'")

    return df.loc[mask].copy()


def save_compressed_fingerprints(path, X, canonical_smiles=None):
    arrays = {"X": X}
    if canonical_smiles is not None:
        arrays["canonical_smiles"] = np.asarray(canonical_smiles, dtype=str)
    np.savez_compressed(path, **arrays)


def prepare_target(
    target,
    master_path,
    registry_path,
    output_base_dir,
    radius=2,
    n_bits=2048,
    overwrite=False,
):
    registry = load_target_registry(registry_path)

    if target not in registry:
        raise ValueError(
            f"Target '{target}' not found in {registry_path}. "
            f"Available targets: {list(registry)}"
        )

    query = registry[target]["query"]
    match = registry[target].get("match", "contains")

    activity_threshold_nM = registry[target].get(
        "activity_threshold_nM",
        1000,
    )

    output_base_dir = Path(output_base_dir)
    if output_base_dir.resolve() == HISTORICAL_TARGETS_DIR.resolve():
        raise ValueError(
            "Refusing to overwrite historical v0.5.0 target artifacts; "
            "use data/processed/v051/targets"
        )
    target_dir = output_base_dir / target
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Target output already exists: {target_dir}; pass overwrite=True"
        )

    df = pd.read_parquet(master_path)
    target_df = filter_target(df, query, match=match)
    if target_df.empty:
        raise ValueError(f"No rows found for target '{target}' using query '{query}'.")

    target_dir.mkdir(parents=True, exist_ok=True)

    clean_path = target_dir / "clean.parquet"
    X_path = target_dir / "X_morgan.npz"
    y_path = target_dir / "y_classification.npy"
    descriptors_path = target_dir / "descriptors.parquet"
    metadata_path = target_dir / "metadata.json"

    X_valid_structure, valid_structure_indices = featurize_smiles(
        target_df["canonical_smiles"].tolist(),
        radius=radius,
        n_bits=n_bits,
    )

    structure_df = target_df.iloc[valid_structure_indices].copy()
    structure_df["_fingerprint_row"] = np.arange(len(structure_df))

    descriptors = compute_basic_descriptors(
        structure_df["canonical_smiles"].tolist()
    )
    descriptors = descriptors.rename(columns={"index": "_structure_row"})

    affinity_values = pd.to_numeric(
        structure_df["aff_nM_median"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_affinity = np.isfinite(affinity_values) & (affinity_values > 0)
    qed_values = descriptors["QED"].to_numpy(dtype=float)
    valid_qed = np.isfinite(qed_values) & (qed_values >= 0) & (qed_values <= 1)

    # Descriptor calculation uses the same RDKit parsing as fingerprinting. Keep
    # this explicit mapping so every exclusion is applied identically to all files.
    descriptor_rows = descriptors["_structure_row"].to_numpy(dtype=int)
    retained_descriptor_mask = finite_affinity[descriptor_rows] & valid_qed
    descriptors = descriptors.loc[retained_descriptor_mask].reset_index(drop=True)
    retained_structure_rows = descriptors.pop("_structure_row").to_numpy(dtype=int)
    retained_df = structure_df.iloc[retained_structure_rows].copy()
    fingerprint_rows = retained_df.pop("_fingerprint_row").to_numpy(dtype=int)
    X = X_valid_structure[fingerprint_rows]
    affinity_values = retained_df["aff_nM_median"].to_numpy(dtype=float)
    canonical_smiles = descriptors["canonical_smiles"].to_numpy(dtype=str)

    # Normalize the retained table identity to the exact canonical form stored
    # with the descriptors and fingerprints.
    retained_df["canonical_smiles"] = canonical_smiles

    y = (affinity_values <= activity_threshold_nM).astype(int)
    # The master-table label uses a generic threshold. Target artifacts must
    # consistently use the target-specific threshold recorded in the registry.
    retained_df["label"] = y
    retained_df.to_parquet(clean_path, index=False)
    save_compressed_fingerprints(X_path, X, canonical_smiles)
    np.save(y_path, y)

    p_affinity = 9 - np.log10(affinity_values) # pAffinity for regression tasks
    affinity_path = target_dir / "affinity.npy"
    np.save(affinity_path, p_affinity)
    descriptors.insert(0, "index", np.arange(len(descriptors), dtype=int))
    descriptors["label"] = y
    descriptors.to_parquet(descriptors_path, index=False)

    metadata = {
        "target": target,
        "source_master_path": str(master_path),
        "source_master_sha256": file_sha256(master_path),
        "query": query,
        "n_molecules_raw": int(len(target_df)),
        "n_molecules_valid": int(len(y)),
        "exclusions": {
            "invalid_structure_or_fingerprint": int(
                len(target_df) - len(valid_structure_indices)
            ),
            "non_finite_or_non_positive_affinity": int((~finite_affinity).sum()),
            "invalid_qed": int(
                (finite_affinity[descriptor_rows] & ~valid_qed).sum()
            ),
        },
        "active_fraction": float(y.mean()),
        "fingerprint": {
            "type": "Morgan",
            "radius": radius,
            "n_bits": n_bits,
        },
        "activity_threshold_nM": activity_threshold_nM,
        "match": match,
        "descriptor_set": "basic_rdkit_2d",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "clean": str(clean_path),
            "X_morgan": str(X_path),
            "y_classification": str(y_path),
            "descriptors": str(descriptors_path),
            "affinity": str(affinity_path),
        },
        "regression_target": "pAffinity",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--master-path",
        default="data/processed/v051/bindingdb_clean_all.parquet",
    )
    parser.add_argument(
        "--registry-path",
        default="configs/targets.yaml",
    )
    parser.add_argument(
        "--output-base-dir",
        default="data/processed/v051/targets",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    prepare_target(
        target=args.target,
        master_path=args.master_path,
        registry_path=args.registry_path,
        output_base_dir=args.output_base_dir,
        radius=args.radius,
        n_bits=args.n_bits,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
