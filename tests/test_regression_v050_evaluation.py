import json

import numpy as np
import pytest

from src.data.targets import load_target_regression_evaluation
from src.regression.evaluation import (
    bemis_murcko_scaffolds,
    count_distinct_scaffolds,
    cumulative_curve_auc,
    evaluate_acquired_molecules,
    joint_discovery_mask,
    mean_pairwise_tanimoto_distance,
)
from src.regression.simulation import run_regression_simulation


def test_activity_and_qed_thresholds_are_inclusive_and_direction_is_nm_lower():
    affinity_nm = np.array([499.9, 500.0, 500.1, 100.0])
    qed = np.array([0.6, 0.6, 1.0, 0.5999])

    assert joint_discovery_mask(affinity_nm, qed).tolist() == [True, True, False, False]


def test_cumulative_joint_discovery_and_trapezoidal_auc():
    measurements = [0, 10, 20, 30]
    discoveries = [0, 2, 5, 5]

    assert cumulative_curve_auc(measurements, discoveries) == pytest.approx(95.0)


def test_scaffold_count_uses_distinct_nonempty_bemis_murcko_scaffolds():
    scaffolds = bemis_murcko_scaffolds(["c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CCO"])

    assert scaffolds[0] == scaffolds[1]
    assert scaffolds[3] == ""
    assert count_distinct_scaffolds(scaffolds) == 2


def test_mean_pairwise_tanimoto_distance_and_small_campaign_treatment():
    fingerprints = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )

    # Pair distances are 0, 1, and 1.
    assert mean_pairwise_tanimoto_distance(fingerprints) == pytest.approx(2 / 3)
    assert np.isnan(mean_pairwise_tanimoto_distance(fingerprints[:1]))
    assert np.isnan(mean_pairwise_tanimoto_distance(fingerprints[:0]))


def test_evaluation_counts_only_experimental_qualifying_acquired_molecules():
    affinity_nm = np.array([500.0, 100.0, 501.0, 20.0])
    qed = np.array([0.6, 0.59, 0.9, 0.8])
    fingerprints = np.eye(4, dtype=np.int8)
    scaffolds = np.array(["c1ccccc1", "", "C1CCCCC1", "c1ccccc1"])

    result = evaluate_acquired_molecules(
        acquired_indices=[0, 1, 2, 3],
        affinity_nm=affinity_nm,
        qed=qed,
        fingerprints=fingerprints,
        scaffolds=scaffolds,
    )

    assert result["affinity_discovery_count"] == 3
    assert result["joint_discovery_count"] == 2
    assert result["joint_scaffold_count"] == 1
    assert result["joint_qualifying_indices"].tolist() == [0, 3]


def test_egfr_evaluation_arrays_are_molecularly_aligned():
    X, y, qed, affinity_nm, smiles = load_target_regression_evaluation("EGFR")

    assert len(X) == len(y) == len(qed) == len(affinity_nm) == len(smiles) == 14174
    assert np.allclose(y, 9.0 - np.log10(affinity_nm))
    assert joint_discovery_mask(affinity_nm, qed).sum() > 0


def test_simulation_history_excludes_initial_and_test_molecules_from_discovery():
    rng = np.random.default_rng(19)
    X = rng.integers(0, 2, size=(50, 16), dtype=np.int8)
    y = rng.normal(size=50)
    qed = np.full(50, 0.8)
    affinity_nm = np.full(50, 100.0)
    smiles = np.array(["CC"] * 50)
    scaffolds = np.array([""] * 50)

    history = run_regression_simulation(
        X,
        y,
        strategy="ucb_qed",
        qed=qed,
        affinity_nm=affinity_nm,
        canonical_smiles=smiles,
        scaffolds=scaffolds,
        model_name="extra_trees",
        random_state=4,
        n_initial=20,
        batch_size=10,
        n_rounds=2,
    )

    assert history["initial_joint_discovery_count"].tolist() == [20, 20, 20]
    assert history["joint_discovery_count"].tolist() == [0, 10, 20]
    assert history["affinity_discovery_count"].tolist() == [0, 10, 20]
    assert history["n_cumulative_acquired"].tolist() == [0, 10, 20]
    assert history.iloc[-1]["joint_discovery_count"] == 20

    final_acquired = set(json.loads(history.iloc[-1]["cumulative_acquired_indices"]))
    test_indices = set(history.attrs["test_indices"].tolist())
    initial_indices = set(history.attrs["initial_labeled_indices"].tolist())
    assert final_acquired.isdisjoint(test_indices)
    assert final_acquired.isdisjoint(initial_indices)

    curve = history["joint_discovery_count"].to_numpy()
    x = history["n_cumulative_acquired"].to_numpy()
    assert history.iloc[-1]["joint_discovery_auc"] == cumulative_curve_auc(x, curve)
