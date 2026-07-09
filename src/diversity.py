import numpy as np


def select_diverse_subset(X, candidate_indices, n_select, random_state=42):
    """
    Select a diverse subset using greedy farthest-point sampling.

    Scientific meaning
    ------------------
    Select molecules that cover the candidate feature space as broadly
    as possible.

    Algorithm
    ---------
    1. Pick one candidate at random.
    2. Compute each candidate's distance to the selected set.
    3. Iteratively select the candidate whose closest selected neighbor
       is farthest away.
    """

    if n_select > len(candidate_indices):
        raise ValueError("n_select cannot exceed number of candidate indices.")

    rng = np.random.default_rng(random_state)

    candidate_indices = np.asarray(candidate_indices)

    first_index = rng.choice(candidate_indices)
    selected = [first_index]

    min_distances = np.linalg.norm(
        X[candidate_indices] - X[first_index],
        axis=1,
    )

    for _ in range(1, n_select):
        already_selected_mask = np.isin(candidate_indices, selected)
        min_distances[already_selected_mask] = -np.inf

        next_position = np.argmax(min_distances)
        next_index = candidate_indices[next_position]

        selected.append(next_index)

        distances_to_new = np.linalg.norm(
            X[candidate_indices] - X[next_index],
            axis=1,
        )

        min_distances = np.minimum(min_distances, distances_to_new)

    return np.array(selected, dtype=int)