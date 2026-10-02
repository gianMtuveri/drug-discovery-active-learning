from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.partitions import generate_partition_manifest
from src.data.representations import make_molecule_ids
from src.regression.base import RegressionSurrogate
from src.regression.simulation import run_regression_simulation


MODELS = ("random_forest", "extra_trees", "bayesian_ridge", "gaussian_process")
STRATEGIES = ("ucb", "ucb_diversity")


def _fixture(n=52):
    rng = np.random.default_rng(81)
    X_morgan = rng.integers(0, 2, size=(n, 16), dtype=np.uint8)
    X_model = rng.normal(size=(n, 5))
    y = 6.0 + 0.5 * X_model[:, 0] - 0.25 * X_model[:, 1]
    affinity_nm = 10 ** (9.0 - y)
    qed = np.full(n, 0.7)
    smiles = np.asarray(["CC"] * n)
    scaffolds = np.asarray([""] * n)
    ids = make_molecule_ids([f"C{'C' * i}" for i in range(1, n + 1)])
    manifest = generate_partition_manifest(ids, random_seed=0, n_initial=20)
    return X_model, X_morgan, y, affinity_nm, qed, smiles, scaffolds, ids, manifest


def _run_legacy_or_integrated(model_name, strategy, integrated):
    X_model, X_morgan, y, affinity_nm, qed, smiles, scaffolds, ids, manifest = _fixture()
    common = dict(
        y=y, strategy=strategy, qed=qed, affinity_nm=affinity_nm,
        canonical_smiles=smiles, scaffolds=scaffolds, model_name=model_name,
        beta=2.0, random_state=0, n_initial=20, batch_size=10,
        n_rounds=1, test_size=0.2,
    )
    if integrated:
        return run_regression_simulation(
            None, X_model=X_morgan, X_morgan=X_morgan,
            representation_name="morgan", partition_manifest=manifest,
            molecule_ids=ids, **common,
        )
    return run_regression_simulation(X_morgan, **common)


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_integrated_morgan_active_learning_matches_legacy(model_name, strategy):
    legacy = _run_legacy_or_integrated(model_name, strategy, integrated=False)
    integrated = _run_legacy_or_integrated(model_name, strategy, integrated=True)

    for attribute in (
        "pool_indices", "test_indices", "initial_labeled_indices", "acquired_batches"
    ):
        old = legacy.attrs[attribute]
        new = integrated.attrs[attribute]
        if isinstance(old, list):
            assert len(old) == len(new)
            for old_batch, new_batch in zip(old, new):
                np.testing.assert_array_equal(old_batch, new_batch)
        else:
            np.testing.assert_array_equal(old, new)
    comparable_columns = [
        column for column in legacy.columns if column != "round_runtime_seconds"
    ]
    pd.testing.assert_frame_equal(
        legacy[comparable_columns], integrated[comparable_columns],
        check_exact=False, rtol=1e-14, atol=1e-14,
    )


class _RecordingSurrogate(RegressionSurrogate):
    def __init__(self, record):
        self.record = record

    def fit(self, X, y):
        self.record["fit_dimensions"].append(X.shape[1])
        return self

    def predict(self, X):
        self.record["predict_dimensions"].append(X.shape[1])
        return np.asarray(X[:, 0], dtype=float)

    def predict_with_uncertainty(self, X):
        self.record["predict_dimensions"].append(X.shape[1])
        return np.asarray(X[:, 0], dtype=float), np.zeros(len(X), dtype=float)


def _isolation_run(monkeypatch, X_morgan, strategy="ucb"):
    import src.regression.simulation as simulation

    n = len(X_morgan)
    X_model = np.column_stack([
        np.arange(n, dtype=float),
        np.linspace(-1.0, 1.0, n),
        np.cos(np.arange(n)),
    ])
    y = 0.1 * np.arange(n, dtype=float)
    ids = make_molecule_ids([f"N{'C' * i}" for i in range(1, n + 1)])
    manifest = generate_partition_manifest(ids, random_seed=0, n_initial=20)
    record = {"fit_dimensions": [], "predict_dimensions": []}
    monkeypatch.setattr(
        simulation, "make_representation_surrogate",
        lambda **kwargs: _RecordingSurrogate(record),
    )
    history = simulation.run_regression_simulation(
        None, y, strategy, X_model=X_model, X_morgan=X_morgan,
        representation_name="rdkit_descriptors", partition_manifest=manifest,
        molecule_ids=ids, model_name="random_forest", random_state=0,
        n_initial=20, batch_size=10, n_rounds=1, beta=2.0,
    )
    return history, record


def test_surrogate_uses_x_model_not_morgan(monkeypatch):
    rng = np.random.default_rng(3)
    X_morgan = rng.integers(0, 2, size=(60, 19), dtype=np.uint8)
    history, record = _isolation_run(monkeypatch, X_morgan, strategy="ucb")

    assert record["fit_dimensions"] == [3, 3]
    assert record["predict_dimensions"] and set(record["predict_dimensions"]) == {3}
    assert history.attrs["model_feature_dimension"] == 3
    assert history.attrs["morgan_feature_dimension"] == 19


def test_changing_morgan_does_not_change_nondiversity_predictions_or_selection(monkeypatch):
    rng = np.random.default_rng(7)
    first, _ = _isolation_run(
        monkeypatch, rng.integers(0, 2, size=(60, 12), dtype=np.uint8), "ucb"
    )
    second, _ = _isolation_run(
        monkeypatch, rng.integers(0, 2, size=(60, 27), dtype=np.uint8), "ucb"
    )
    np.testing.assert_array_equal(
        first.attrs["acquired_batches"][0], second.attrs["acquired_batches"][0]
    )
    for column in ("selected_mean_prediction", "selected_mean_uncertainty"):
        np.testing.assert_allclose(first[column], second[column], equal_nan=True)


def test_diversity_selection_uses_morgan_and_isolated_from_x_model(monkeypatch):
    rng = np.random.default_rng(14)
    first_morgan = rng.integers(0, 2, size=(60, 32), dtype=np.uint8)
    second_morgan = np.zeros((60, 32), dtype=np.uint8)
    second_morgan[:, 0] = 1
    second_morgan[np.arange(60), np.arange(60) % 31 + 1] = 1

    first, _ = _isolation_run(monkeypatch, first_morgan, "ucb_diversity")
    second, _ = _isolation_run(monkeypatch, second_morgan, "ucb_diversity")
    first_batch = first.attrs["acquired_batches"][0]
    second_batch = second.attrs["acquired_batches"][0]

    assert first_batch[0] == second_batch[0]  # same surrogate-score winner
    assert not np.array_equal(first_batch[1:], second_batch[1:])


def test_representation_aware_boundary_rejects_row_mismatch_and_hash_mismatch():
    X_model, X_morgan, y, _, _, _, _, ids, manifest = _fixture()
    with pytest.raises(ValueError, match="same number of molecules"):
        run_regression_simulation(
            None, y, "ucb", X_model=X_model[:-1], X_morgan=X_morgan,
            representation_name="rdkit_descriptors", partition_manifest=manifest,
            molecule_ids=ids, random_state=0,
        )
    reordered = ids.copy()
    reordered[[0, 1]] = reordered[[1, 0]]
    with pytest.raises(ValueError, match="identity or ordering"):
        run_regression_simulation(
            None, y, "ucb", X_model=X_model, X_morgan=X_morgan,
            representation_name="rdkit_descriptors", partition_manifest=manifest,
            molecule_ids=reordered, random_state=0,
        )
