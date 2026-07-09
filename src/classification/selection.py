import numpy as np
from src.diversity import select_diverse_subset

def select_random(unlabeled_indices, batch_size=10, random_state=42):
    """
    Select molecules randomly from the unlabeled pool.
    This is the baseline strategy.
    """

    rng = np.random.default_rng(random_state)

    return rng.choice(
        unlabeled_indices,
        size=batch_size,
        replace=False,
    )


def select_greedy(unlabeled_indices, probabilities, batch_size=10):
    """
    Select molecules with the highest predicted probability of being active.
    This is exploitation.
    """

    order = np.argsort(probabilities)[::-1]

    selected_positions = order[:batch_size]

    return unlabeled_indices[selected_positions]


def select_uncertainty_topk(unlabeled_indices, probabilities, batch_size=10):
    """
    Select the k most uncertain molecules.

    Scientific meaning
    ------------------
    This strategy queries molecules closest to the current decision boundary,
    where the model has the lowest confidence.

    Implementation
    --------------
    Uncertainty is measured as distance from P(active)=0.5.

    The method greedily returns the top-k most uncertain molecules, without
    considering whether selected molecules are similar to each other.
    """

    uncertainty = np.abs(probabilities - 0.5)

    order = np.argsort(uncertainty)

    selected_positions = order[:batch_size]

    return unlabeled_indices[selected_positions]


def select_uncertainty_diverse(
    X_pool,
    unlabeled_indices,
    probabilities,
    batch_size=10,
    candidate_pool_size=100,
    random_state=42,
):
    """
    Select uncertain molecules while encouraging diversity.

    Scientific meaning
    ------------------
    First identify molecules close to the model decision boundary.
    Then, among those uncertain candidates, select a diverse subset.

    This aims to reduce redundancy in pure uncertainty sampling.

    Strategy
    --------
    1. Rank unlabeled molecules by uncertainty.
    2. Keep the most uncertain candidate_pool_size molecules.
    3. Select batch_size diverse molecules from this uncertain candidate pool.
    """

    if candidate_pool_size < batch_size:
        raise ValueError("candidate_pool_size must be >= batch_size.")

    candidate_pool_size = min(candidate_pool_size, len(unlabeled_indices))

    uncertainty = np.abs(probabilities - 0.5)

    order = np.argsort(uncertainty)

    candidate_positions = order[:candidate_pool_size]
    candidate_indices = unlabeled_indices[candidate_positions]

    selected_indices = select_diverse_subset(
        X=X_pool,
        candidate_indices=candidate_indices,
        n_select=batch_size,
        random_state=random_state,
    )

    return selected_indices


def select_by_score(
    unlabeled_indices,
    scores,
    batch_size=10,
    highest=True,
):
    """
    Select molecules according to a generic score.

    Parameters
    ----------
    unlabeled_indices : np.ndarray
        Candidate molecule indices.
    scores : np.ndarray
        One score per unlabeled molecule.
    batch_size : int
        Number of molecules to select.
    highest : bool
        If True, select highest scores.
        If False, select lowest scores.
    """

    order = np.argsort(scores)

    if highest:
        order = order[::-1]

    selected_positions = order[:batch_size]

    return unlabeled_indices[selected_positions]


def select_query_by_committee(
    unlabeled_indices,
    disagreement_scores,
    batch_size=10,
):
    """
    Select molecules with the highest committee disagreement.

    Scientific meaning
    ------------------
    This strategy selects molecules for which different models strongly
    disagree about P(active).

    The disagreement score is computed outside this function.
    """

    return select_by_score(
        unlabeled_indices=unlabeled_indices,
        scores=disagreement_scores,
        batch_size=batch_size,
        highest=True,
    )


def select_query_by_committee(
    unlabeled_indices,
    disagreement_scores,
    batch_size=10,
):
    """
    Select molecules with the highest committee disagreement.

    Scientific meaning
    ------------------
    Molecules are selected when different models strongly disagree
    about their probability of being active.
    """

    return select_by_score(
        unlabeled_indices=unlabeled_indices,
        scores=disagreement_scores,
        batch_size=batch_size,
        highest=True,
    )