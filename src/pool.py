import numpy as np


def initialize_diverse(X, n_initial=20, random_state=42):
    """
    Select an initial labeled set that is diverse in feature space.

    Method
    ------
    Greedy farthest-point sampling:

    1. Choose one random molecule.
    2. At each step, choose the molecule whose closest distance to the
       already selected set is as large as possible.

    Scientific meaning
    ------------------
    This simulates an initial experimental design that tries to cover
    the available chemical/feature space instead of sampling randomly.
    """

    if n_initial >= len(X):
        raise ValueError("n_initial must be smaller than the number of samples.")

    rng = np.random.default_rng(random_state)

    all_indices = np.arange(len(X))

    first_index = rng.choice(all_indices)
    selected = [first_index]

    min_distances = np.linalg.norm(X - X[first_index], axis=1)

    for _ in range(1, n_initial):
        min_distances[selected] = -np.inf

        next_index = np.argmax(min_distances)
        selected.append(next_index)

        distances_to_new = np.linalg.norm(X - X[next_index], axis=1)

        min_distances = np.minimum(min_distances, distances_to_new)

    labeled_indices = np.array(selected, dtype=int)

    unlabeled_indices = np.setdiff1d(
        all_indices,
        labeled_indices,
    )

    return labeled_indices, unlabeled_indices


def initialize_random(n_samples, n_initial=20, random_state=42):
    """
    Randomly select the initial experimentally tested molecules.

    Scientific meaning
    ------------------
    This represents an initial experimental campaign where the starting
    molecules are sampled without using any structural or feature-space
    diversity criterion.
    """

    if n_initial >= n_samples:
        raise ValueError("n_initial must be smaller than n_samples.")

    rng = np.random.default_rng(random_state)

    all_indices = np.arange(n_samples)

    labeled_indices = rng.choice(
        all_indices,
        size=n_initial,
        replace=False,
    )

    unlabeled_indices = np.setdiff1d(
        all_indices,
        labeled_indices,
    )

    return labeled_indices, unlabeled_indices


def initialize_pool(
    X,
    n_initial=20,
    strategy="random",
    random_state=42,
):
    """
    Initialize the labeled and unlabeled pools.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for the training pool.
    n_initial : int
        Number of molecules initially labeled.
    strategy : str
        Initialization strategy.
    random_state : int
        Random seed.

    Returns
    -------
    labeled_indices : np.ndarray
        Initial labeled molecules.
    unlabeled_indices : np.ndarray
        Remaining unlabeled molecules.
    """

    if strategy == "random":
        return initialize_random(
            n_samples=len(X),
            n_initial=n_initial,
            random_state=random_state,
        )

    if strategy == "diverse":
        return initialize_diverse(
            X=X,
            n_initial=n_initial,
            random_state=random_state,
        )

    raise ValueError(
        "Unknown initialization strategy. "
        "Available strategies: 'random', 'diverse'."
    )


def update_pool(labeled_indices, unlabeled_indices, selected_indices):
    """
    Move selected molecules from the unlabeled pool to the labeled pool.
    """

    new_labeled_indices = np.concatenate(
        [labeled_indices, selected_indices]
    )

    new_unlabeled_indices = np.setdiff1d(
        unlabeled_indices,
        selected_indices,
    )

    return new_labeled_indices, new_unlabeled_indices