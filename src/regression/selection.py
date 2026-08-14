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