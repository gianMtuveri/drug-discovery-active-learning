import numpy as np

from src.active_learning.diversity import select_diverse_subset


def robust_scale(values: np.ndarray) -> np.ndarray:
    """
    Robustly center and scale a one-dimensional array.

    Scaling is performed using the median and interquartile range:

        scaled = (x - median(x)) / IQR(x)

    If the IQR is effectively zero, standard deviation is used as a
    fallback. If both are zero, an array of zeros is returned.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("robust_scale expects a one-dimensional array.")

    median = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25

    if np.isclose(iqr, 0.0):
        scale = np.std(values)

        if np.isclose(scale, 0.0):
            return np.zeros_like(values, dtype=float)
    else:
        scale = iqr

    return (values - median) / scale


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize one component, returning zero for no-information ties."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("minmax_normalize expects a one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("minmax_normalize requires finite values.")
    minimum = np.min(values)
    value_range = np.max(values) - minimum
    if np.isclose(value_range, 0.0):
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / value_range


def select_random(
    unlabeled_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.choice(
        unlabeled_indices,
        size=min(batch_size, len(unlabeled_indices)),
        replace=False,
    )


def select_greedy(
    unlabeled_indices: np.ndarray,
    predicted_affinity: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    order = np.argsort(predicted_affinity, kind="stable")[::-1]

    return unlabeled_indices[order[:batch_size]]


def select_uncertainty(
    unlabeled_indices: np.ndarray,
    uncertainty: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    order = np.argsort(uncertainty, kind="stable")[::-1]

    return unlabeled_indices[order[:batch_size]]


def select_ucb(
    unlabeled_indices: np.ndarray,
    predicted_affinity: np.ndarray,
    uncertainty: np.ndarray,
    batch_size: int,
    beta: float,
) -> np.ndarray:
    """
    Select molecules using robustly normalized Upper Confidence Bound.

    score(x) = robust_scale(mu(x)) + beta * robust_scale(sigma(x))

    Parameters
    ----------
    unlabeled_indices
        Global indices of currently unlabeled molecules.
    predicted_affinity
        Predicted mean pAffinity for the unlabeled molecules.
    uncertainty
        Predictive standard deviation for the unlabeled molecules.
    batch_size
        Number of molecules to acquire.
    beta
        Weight assigned to predictive uncertainty.

        beta = 0:
            Pure greedy acquisition.

        increasing beta:
            Increasing exploration pressure.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    if len(unlabeled_indices) != len(predicted_affinity):
        raise ValueError(
            "unlabeled_indices and predicted_affinity must have equal length."
        )

    if len(unlabeled_indices) != len(uncertainty):
        raise ValueError(
            "unlabeled_indices and uncertainty must have equal length."
        )

    predicted_scaled = robust_scale(predicted_affinity)
    uncertainty_scaled = robust_scale(uncertainty)

    scores = predicted_scaled + beta * uncertainty_scaled

    order = np.argsort(scores, kind="stable")[::-1]

    return unlabeled_indices[order[:batch_size]]


def normalized_ucb_scores(
    predicted_affinity: np.ndarray,
    uncertainty: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    """Return v0.5 UCB scores after final candidate-pool min-max scaling.

    The pre-existing UCB calculation is deliberately retained: prediction
    and uncertainty are each robust-scaled before being combined.  Min-max
    scaling the final score does not alter its ranking.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    predicted_affinity = np.asarray(predicted_affinity, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if predicted_affinity.ndim != 1 or uncertainty.ndim != 1:
        raise ValueError("UCB inputs must be one-dimensional.")
    if len(predicted_affinity) != len(uncertainty):
        raise ValueError("UCB inputs must have equal length.")

    raw_ucb = robust_scale(predicted_affinity) + beta * robust_scale(uncertainty)
    return minmax_normalize(raw_ucb)


def acquisition_scores(
    predicted_affinity: np.ndarray,
    uncertainty: np.ndarray,
    qed: np.ndarray | None = None,
    beta: float = 2.0,
) -> np.ndarray:
    """Calculate normalized UCB, optionally augmented by raw RDKit QED."""
    scores = normalized_ucb_scores(predicted_affinity, uncertainty, beta)
    if qed is None:
        return scores

    qed = np.asarray(qed, dtype=float)
    if qed.ndim != 1 or len(qed) != len(scores):
        raise ValueError("QED must be one-dimensional and aligned with UCB scores.")
    return scores + 0.5 * qed


def select_acquisition_topk(
    unlabeled_indices: np.ndarray,
    scores: np.ndarray,
    batch_size: int = 10,
) -> np.ndarray:
    """Select by score using the pipeline's existing stable-reverse ordering."""
    unlabeled_indices = np.asarray(unlabeled_indices)
    scores = np.asarray(scores, dtype=float)
    if len(unlabeled_indices) != len(scores):
        raise ValueError("Candidate indices and scores must have equal length.")
    order = np.argsort(scores, kind="stable")[::-1]
    return unlabeled_indices[order[: min(batch_size, len(order))]]


def select_acquisition_diverse(
    X: np.ndarray,
    unlabeled_indices: np.ndarray,
    labeled_indices: np.ndarray,
    scores: np.ndarray,
    batch_size: int = 10,
    candidate_pool_size: int = 50,
) -> np.ndarray:
    """Greedily maximize minimum Tanimoto distance within a scored shortlist.

    The first molecule is the highest-scoring candidate.  Remaining ties are
    resolved by shortlist order (score order, with the legacy reverse-stable
    convention for exact score ties), making selection deterministic.
    """
    X = np.asarray(X)
    unlabeled_indices = np.asarray(unlabeled_indices, dtype=int)
    labeled_indices = np.asarray(labeled_indices, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(unlabeled_indices) != len(scores):
        raise ValueError("Candidate indices and scores must have equal length.")
    if batch_size <= 0 or candidate_pool_size <= 0:
        raise ValueError("Batch and candidate-pool sizes must be positive.")

    order = np.argsort(scores, kind="stable")[::-1]
    shortlist = unlabeled_indices[order[: min(candidate_pool_size, len(order))]]
    n_select = min(batch_size, len(shortlist))
    if n_select == 0:
        return np.array([], dtype=int)

    selected = [int(shortlist[0])]
    while len(selected) < n_select:
        remaining = shortlist[~np.isin(shortlist, selected)]
        references = np.concatenate([labeled_indices, np.asarray(selected)])

        candidate_fp = X[remaining].astype(bool, copy=False)
        reference_fp = X[references].astype(bool, copy=False)
        intersections = candidate_fp.astype(np.uint16) @ reference_fp.T.astype(np.uint16)
        candidate_counts = candidate_fp.sum(axis=1)[:, None]
        reference_counts = reference_fp.sum(axis=1)[None, :]
        similarities = intersections / (candidate_counts + reference_counts - intersections)
        min_distances = 1.0 - similarities.max(axis=1)
        selected.append(int(remaining[np.argmax(min_distances)]))

    return np.asarray(selected, dtype=int)


def select_uncertainty_diverse(
    X: np.ndarray,
    unlabeled_indices: np.ndarray,
    uncertainty: np.ndarray,
    batch_size: int,
    candidate_pool_size: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Select a diverse subset from the most uncertain molecules.
    """
    n_candidates = min(
        candidate_pool_size,
        len(unlabeled_indices),
    )

    candidate_order = (
        np.argsort(uncertainty, kind="stable")[::-1][:n_candidates]
    )

    candidate_indices = unlabeled_indices[candidate_order]

    return select_diverse_subset(
        X=X,
        candidate_indices=candidate_indices,
        n_select=min(batch_size, len(candidate_indices)),
        random_state=random_state,
    )
