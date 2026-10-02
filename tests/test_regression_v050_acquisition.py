import numpy as np
import pytest

from src.data.targets import load_target_regression
from src.regression.selection import (
    acquisition_scores,
    minmax_normalize,
    normalized_ucb_scores,
    select_acquisition_diverse,
    select_acquisition_topk,
    select_ucb,
)
from src.regression.simulation import run_regression_simulation


STRATEGIES = [
    "ucb",
    "ucb_qed",
    "ucb_diversity",
    "ucb_qed_diversity",
]
MODELS = [
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
]


def test_constant_component_minmax_normalizes_to_finite_zeros():
    normalized = minmax_normalize(np.full(7, 3.25))

    np.testing.assert_array_equal(normalized, np.zeros(7))
    assert np.isfinite(normalized).all()


def test_nonconstant_minmax_normalization_is_unchanged():
    values = np.array([-2.0, 0.0, 6.0, 2.0])
    expected = (values - values.min()) / (values.max() - values.min())

    np.testing.assert_array_equal(minmax_normalize(values), expected)


def test_constant_pure_ucb_preserves_reverse_stable_tie_order():
    indices = np.array([4, 8, 15, 16, 23, 42])
    scores = normalized_ucb_scores(np.ones(6), np.ones(6))

    np.testing.assert_array_equal(scores, np.zeros(6))
    np.testing.assert_array_equal(
        select_acquisition_topk(indices, scores, batch_size=3),
        np.array([42, 23, 16]),
    )


def test_constant_ucb_leaves_qed_to_determine_composite_ranking():
    indices = np.arange(4)
    qed = np.array([0.2, 0.9, 0.5, 0.1])
    combined = acquisition_scores(np.ones(4), np.ones(4), qed=qed)

    np.testing.assert_array_equal(combined, 0.5 * qed)
    np.testing.assert_array_equal(
        select_acquisition_topk(indices, combined, batch_size=4),
        np.array([1, 2, 0, 3]),
    )


def test_normalized_ucb_topk_preserves_original_candidate_ranking():
    indices = np.arange(12)
    mean = np.array([1.2, 0.1, 2.4, 1.0, 3.1, 2.0, 0.7, 2.8, 1.8, 0.4, 1.4, 2.2])
    uncertainty = np.array([0.2, 1.1, 0.3, 0.8, 0.1, 0.7, 1.4, 0.4, 0.6, 1.0, 0.9, 0.5])

    original = select_ucb(indices, mean, uncertainty, batch_size=10, beta=2.0)
    normalized = select_acquisition_topk(
        indices,
        acquisition_scores(mean, uncertainty, beta=2.0),
        batch_size=10,
    )

    assert np.array_equal(normalized, original)


def test_qed_changes_ranking_using_unscaled_half_qed_formula():
    mean = np.array([0.0, 0.1, 1.0, 0.8])
    uncertainty = np.array([0.1, 0.2, 0.4, 0.3])
    qed = np.array([1.0, 0.0, 0.0, 0.0])
    ucb = acquisition_scores(mean, uncertainty)
    combined = acquisition_scores(mean, uncertainty, qed=qed)

    assert np.allclose(combined, ucb + 0.5 * qed)
    assert np.argsort(combined)[::-1].tolist() != np.argsort(ucb)[::-1].tolist()


def _tanimoto_distance(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    return 1.0 - np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def test_diverse_batch_is_distinct_top50_and_farthest_from_labelled_and_batch():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(65, 32), dtype=np.int8)
    labeled = np.arange(5)
    candidates = np.arange(5, 65)
    scores = np.linspace(0.0, 1.0, len(candidates))

    selected = select_acquisition_diverse(
        X, candidates, labeled, scores, batch_size=10, candidate_pool_size=50
    )
    top50 = set(select_acquisition_topk(candidates, scores, batch_size=50))

    assert len(selected) == len(set(selected)) == 10
    assert set(selected).issubset(top50)
    assert selected[0] == select_acquisition_topk(candidates, scores, batch_size=1)[0]

    shortlist = select_acquisition_topk(candidates, scores, batch_size=50)
    for position in range(1, len(selected)):
        remaining = [i for i in shortlist if i not in selected[:position]]
        references = np.concatenate([labeled, selected[:position]])
        distances = [
            min(_tanimoto_distance(X[candidate], X[reference]) for reference in references)
            for candidate in remaining
        ]
        assert selected[position] == remaining[int(np.argmax(distances))]


def test_qed_loader_and_simulation_indices_stay_aligned_and_exclude_test_set():
    X, y, qed = load_target_regression("EGFR", include_qed=True)
    assert X.shape[0] == y.shape[0] == qed.shape[0] == 14174

    subset = np.arange(80)
    history = run_regression_simulation(
        X[subset], y[subset], qed=qed[subset], strategy="ucb_qed_diversity",
        model_name="extra_trees", random_state=3, n_rounds=1,
    )
    acquired = np.concatenate(history.attrs["acquired_batches"])
    test = history.attrs["test_indices"]

    assert not np.intersect1d(acquired, test).size
    assert np.array_equal(qed[subset][acquired], qed[acquired])
    assert np.array_equal(X[subset][acquired], X[acquired])


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("model_name", MODELS)
def test_every_surrogate_executes_one_round_with_every_strategy(strategy, model_name):
    rng = np.random.default_rng(11)
    X = rng.integers(0, 2, size=(48, 12), dtype=np.int8)
    y = rng.normal(size=48)
    qed = rng.uniform(size=48)

    history = run_regression_simulation(
        X, y, qed=qed, strategy=strategy, model_name=model_name,
        random_state=5, n_initial=20, batch_size=10, n_rounds=1,
    )

    assert history["round"].tolist() == [0, 1]
    assert history["n_labeled"].tolist() == [20, 30]
    assert len(history.attrs["acquired_batches"][0]) == 10
