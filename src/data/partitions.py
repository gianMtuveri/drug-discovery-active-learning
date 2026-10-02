"""Deterministic active-learning population partition manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.representations import ordered_population_hash


PARTITION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PartitionManifest:
    schema_version: int
    ordered_population_hash: str
    population_size: int
    random_seed: int
    acquisition_pool_indices: tuple[int, ...]
    held_out_test_indices: tuple[int, ...]
    initial_labeled_indices: tuple[int, ...]
    test_size: float
    n_initial: int
    split_method: str = "sklearn.model_selection.train_test_split"
    initial_selection_method: str = "numpy.random.default_rng.choice"


def generate_partition_manifest(
    molecule_ids: Sequence[str],
    *,
    random_seed: int,
    test_size: float = 0.2,
    n_initial: int = 20,
) -> PartitionManifest:
    """Reproduce the v0.5.0 split and initial-labelled selection sequence."""
    ids = np.asarray(molecule_ids)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("Molecular identities must be a non-empty one-dimensional array")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate molecular identities found")
    if n_initial <= 0:
        raise ValueError("n_initial must be positive")

    pool_indices, test_indices = train_test_split(
        np.arange(len(ids)),
        test_size=test_size,
        random_state=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    initial = rng.choice(
        pool_indices,
        size=min(n_initial, len(pool_indices)),
        replace=False,
    )
    manifest = PartitionManifest(
        schema_version=PARTITION_SCHEMA_VERSION,
        ordered_population_hash=ordered_population_hash(ids),
        population_size=len(ids),
        random_seed=random_seed,
        acquisition_pool_indices=tuple(int(value) for value in pool_indices),
        held_out_test_indices=tuple(int(value) for value in test_indices),
        initial_labeled_indices=tuple(int(value) for value in initial),
        test_size=float(test_size),
        n_initial=int(n_initial),
    )
    validate_partition_manifest(manifest, ids)
    return manifest


def validate_partition_manifest(
    manifest: PartitionManifest,
    molecule_ids: Sequence[str],
) -> None:
    ids = np.asarray(molecule_ids)
    if manifest.schema_version != PARTITION_SCHEMA_VERSION:
        raise ValueError("Unsupported partition-manifest schema version")
    if manifest.population_size != len(ids):
        raise ValueError("Partition population size does not match molecular identities")
    if manifest.ordered_population_hash != ordered_population_hash(ids):
        raise ValueError("Partition population identity or ordering does not match")

    pool = np.asarray(manifest.acquisition_pool_indices, dtype=int)
    test = np.asarray(manifest.held_out_test_indices, dtype=int)
    initial = np.asarray(manifest.initial_labeled_indices, dtype=int)
    for name, values in (("pool", pool), ("test", test), ("initial", initial)):
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Partition {name} indices contain duplicates")
        if np.any(values < 0) or np.any(values >= len(ids)):
            raise ValueError(f"Partition {name} indices are out of bounds")
    if np.intersect1d(pool, test).size:
        raise ValueError("Acquisition pool and held-out test set overlap")
    if not np.array_equal(np.sort(np.concatenate([pool, test])), np.arange(len(ids))):
        raise ValueError("Pool and test indices do not partition the population")
    if not np.isin(initial, pool).all():
        raise ValueError("Initial labelled indices must belong to the acquisition pool")
    if len(initial) != min(manifest.n_initial, len(pool)):
        raise ValueError("Initial labelled count does not match manifest parameters")


def save_partition_manifest(manifest: PartitionManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")


def load_partition_manifest(
    path: str | Path,
    molecule_ids: Sequence[str],
) -> PartitionManifest:
    payload = json.loads(Path(path).read_text())
    for key in (
        "acquisition_pool_indices",
        "held_out_test_indices",
        "initial_labeled_indices",
    ):
        payload[key] = tuple(payload[key])
    manifest = PartitionManifest(**payload)
    validate_partition_manifest(manifest, molecule_ids)
    return manifest
