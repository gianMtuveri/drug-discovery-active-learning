import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from matplotlib.units import registry
import numpy as np
import pandas as pd
import yaml

from src.fingerprints import featurize_smiles
from src.descriptors import compute_basic_descriptors


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


def save_compressed_fingerprints(path, X):
    np.savez_compressed(path, X=X)


def prepare_target(
    target,
    master_path,
    registry_path,
    output_base_dir,
    radius=2,
    n_bits=2048,
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

    df = pd.read_parquet(master_path)
    target_df = filter_target(df, query, match= match)

    if target_df.empty:
        raise ValueError(f"No rows found for target '{target}' using query '{query}'.")

    target_dir = Path(output_base_dir) / target
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_path = target_dir / "clean.parquet"
    X_path = target_dir / "X_morgan.npz"
    y_path = target_dir / "y_classification.npy"
    descriptors_path = target_dir / "descriptors.parquet"
    metadata_path = target_dir / "metadata.json"

    target_df.to_parquet(clean_path, index=False)

    X, valid_indices = featurize_smiles(
        target_df["canonical_smiles"].tolist(),
        radius=radius,
        n_bits=n_bits,
    )

    affinity_values = target_df.iloc[valid_indices]["aff_nM_median"].to_numpy()

    y = (affinity_values <= activity_threshold_nM).astype(int)

    save_compressed_fingerprints(X_path, X)
    np.save(y_path, y)

    descriptors = compute_basic_descriptors(
        target_df.iloc[valid_indices]["canonical_smiles"].tolist()
    )
    descriptors["label"] = y
    descriptors.to_parquet(descriptors_path, index=False)

    metadata = {
        "target": target,
        "query": query,
        "n_molecules_raw": int(len(target_df)),
        "n_molecules_valid": int(len(y)),
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
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--master-path",
        default="data/processed/bindingdb_clean_all.parquet",
    )
    parser.add_argument(
        "--registry-path",
        default="configs/targets.yaml",
    )
    parser.add_argument(
        "--output-base-dir",
        default="data/processed/targets",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)

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
    )


if __name__ == "__main__":
    main()