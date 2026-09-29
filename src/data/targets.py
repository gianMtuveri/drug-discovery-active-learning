from pathlib import Path
import numpy as np
import pandas as pd


def get_target_dir(target, base_dir="data/processed/targets"):
    """
    Return the processed-data directory for a target.
    """

    return Path(base_dir) / target


def load_target_classification(target, base_dir="data/processed/targets"):
    target_dir = get_target_dir(target, base_dir)

    X_path = target_dir / "X_morgan.npz"
    y_path = target_dir / "y_classification.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing feature file: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")

    X = np.load(X_path)["X"]
    y = np.load(y_path)

    return X, y


def load_target_table(target, base_dir="data/processed/targets"):
    """
    Load the cleaned molecule table for one target.
    """

    path = get_target_dir(target, base_dir) / "clean.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing table file: {path}")

    return pd.read_parquet(path)


def load_target_descriptors(target, base_dir="data/processed/targets"):
    """
    Load descriptor table for one target.
    """

    path = get_target_dir(target, base_dir) / "descriptors.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing descriptor file: {path}")

    return pd.read_parquet(path)


def load_target_regression(
    target,
    base_dir="data/processed/targets",
    *,
    include_qed=False,
):
    """Load aligned Morgan fingerprints and regression targets.

    Set ``include_qed=True`` to also return the RDKit QED values from
    ``descriptors.parquet``.  The default two-value return is retained for
    compatibility with the v0.4 regression scripts.
    """
    target_dir = get_target_dir(target, base_dir)

    X_path = target_dir / "X_morgan.npz"
    y_path = target_dir / "affinity.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing feature file: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing regression target file: {y_path}")

    fingerprint_data = np.load(X_path)
    X = fingerprint_data["X"]
    y = np.load(y_path)

    if len(X) != len(y):
        raise ValueError(
            "Regression fingerprints and targets are not aligned: "
            f"{len(X)} fingerprints != {len(y)} targets"
        )

    if not include_qed:
        return X, y

    descriptors_path = target_dir / "descriptors.parquet"
    if not descriptors_path.exists():
        raise FileNotFoundError(
            f"Missing descriptor file: {descriptors_path}"
        )

    descriptors = pd.read_parquet(descriptors_path)
    if "QED" not in descriptors:
        raise ValueError("Descriptor table does not contain QED")

    qed = descriptors["QED"].to_numpy(dtype=float)
    if len(qed) != len(X):
        raise ValueError(
            "Regression fingerprints, targets, and QED are not aligned: "
            f"{len(X)} fingerprints, {len(y)} targets, {len(qed)} QED values"
        )
    if "canonical_smiles" not in fingerprint_data.files:
        raise ValueError("X_morgan.npz does not contain canonical_smiles identities")
    if "canonical_smiles" not in descriptors:
        raise ValueError("Descriptor table does not contain canonical_smiles identities")
    if not np.array_equal(
        fingerprint_data["canonical_smiles"].astype(str),
        descriptors["canonical_smiles"].astype(str).to_numpy(),
    ):
        raise ValueError("Canonical SMILES order differs between fingerprints and QED")
    if not (np.isfinite(qed).all() and ((qed >= 0) & (qed <= 1)).all()):
        raise ValueError("Descriptor QED values must be finite and in [0, 1]")

    return X, y, qed


def load_target_regression_evaluation(target, base_dir="data/processed/targets"):
    """Load aligned regression, QED, experimental-affinity, and identity data."""
    X, y, qed = load_target_regression(
        target,
        base_dir,
        include_qed=True,
    )
    table = load_target_table(target, base_dir)
    required = {"canonical_smiles", "aff_nM_median"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Target table is missing evaluation columns: {missing}")

    smiles = table["canonical_smiles"].astype(str).to_numpy()
    affinity_nm = table["aff_nM_median"].to_numpy(dtype=float)
    fingerprint_data = np.load(get_target_dir(target, base_dir) / "X_morgan.npz")
    fingerprint_smiles = fingerprint_data["canonical_smiles"].astype(str)
    if not (
        len(affinity_nm) == len(X)
        and np.array_equal(smiles, fingerprint_smiles)
        and np.isfinite(affinity_nm).all()
        and (affinity_nm > 0).all()
    ):
        raise ValueError("Evaluation affinity and molecular identities are not aligned")

    return X, y, qed, affinity_nm, smiles


def validate_target_alignment(target, base_dir="data/processed/targets"):
    """Validate row counts and canonical-SMILES identity across target artifacts."""
    target_dir = get_target_dir(target, base_dir)
    fingerprint_data = np.load(target_dir / "X_morgan.npz")
    X = fingerprint_data["X"]
    affinity = np.load(target_dir / "affinity.npy")
    descriptors = pd.read_parquet(target_dir / "descriptors.parquet")
    clean = pd.read_parquet(target_dir / "clean.parquet")

    lengths = {
        "X_morgan": len(X),
        "affinity": len(affinity),
        "descriptors": len(descriptors),
        "clean": len(clean),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Target artifact row counts are not aligned: {lengths}")

    if "canonical_smiles" not in fingerprint_data.files:
        raise ValueError("X_morgan.npz does not contain canonical_smiles identities")
    if "canonical_smiles" not in descriptors or "canonical_smiles" not in clean:
        raise ValueError("Target tables do not contain canonical_smiles identities")

    fingerprint_smiles = fingerprint_data["canonical_smiles"].astype(str)
    descriptor_smiles = descriptors["canonical_smiles"].astype(str).to_numpy()
    clean_smiles = clean["canonical_smiles"].astype(str).to_numpy()
    if not (
        np.array_equal(fingerprint_smiles, descriptor_smiles)
        and np.array_equal(fingerprint_smiles, clean_smiles)
    ):
        raise ValueError("Canonical SMILES order differs across target artifacts")

    if not np.isfinite(affinity).all():
        raise ValueError("Regression affinity contains non-finite values")
    if "QED" not in descriptors:
        raise ValueError("Descriptor table does not contain QED")
    qed = descriptors["QED"].to_numpy(dtype=float)
    if not (np.isfinite(qed).all() and ((qed >= 0) & (qed <= 1)).all()):
        raise ValueError("Descriptor QED values must be finite and in [0, 1]")

    return lengths
