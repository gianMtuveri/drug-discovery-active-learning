import numpy as np


def make_toy_dataset(n_samples=1000, random_state=42):
    """
    Create a toy binary classification dataset.

    Each point represents a fake molecule.
    Each molecule has two features.
    The label is 1 if the molecule is in an 'active' region.
    """

    rng = np.random.default_rng(random_state)

    # Fake molecular descriptors
    X = rng.normal(size=(n_samples, 2))

    # Define an artificial active region.
    # Molecules near this region are more likely to be active.
    score = (
        1.5 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.8 * X[:, 0] * X[:, 1]
    )

    probability_active = 1 / (1 + np.exp(-score))

    y = rng.binomial(1, probability_active)

    return X, y