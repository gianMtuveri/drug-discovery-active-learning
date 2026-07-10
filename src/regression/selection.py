import numpy as np

from src.active_learning.diversity import select_diverse_subset


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
    order = np.argsort(predicted_affinity)[::-1]
    return unlabeled_indices[order[:batch_size]]


def select_uncertainty(
    unlabeled_indices: np.ndarray,
    uncertainty: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    order = np.argsort(uncertainty)[::-1]
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
    Select diverse molecules from the most uncertain candidate pool.
    """
    n_candidates = min(candidate_pool_size, len(unlabeled_indices))

    candidate_order = np.argsort(uncertainty)[::-1][:n_candidates]
    candidate_indices = unlabeled_indices[candidate_order]

    selected_local = select_diverse_subset(
        X=X,
        candidate_indices=candidate_indices,
        n_select=min(batch_size, len(candidate_indices)),
        random_state=random_state,
    )

    return selected_local