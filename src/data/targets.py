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


def load_target_regression(target, base_dir="data/processed/targets"):
    target_dir = get_target_dir(target, base_dir)

    X_path = target_dir / "X_morgan.npz"
    y_path = target_dir / "affinity.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing feature file: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing regression target file: {y_path}")

    X = np.load(X_path)["X"]
    y = np.load(y_path)

    return X, y