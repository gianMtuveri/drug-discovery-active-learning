from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np


def committee_predict_proba(committee, X):
    """
    Predict P(active) for each molecule using every committee member.

    Returns
    -------
    probabilities : np.ndarray
        Shape: (n_models, n_molecules)
    """

    probabilities = []

    for model in committee:
        proba = model.predict_proba(X)[:, 1]
        probabilities.append(proba)

    return np.vstack(probabilities)


def compute_disagreement_scores(committee, X):
    """
    Quantify disagreement among committee members.

    For classification, disagreement is measured as the variance of
    predicted P(active) across models.

    Returns
    -------
    disagreement : np.ndarray
        One disagreement score per molecule.
    """

    probabilities = committee_predict_proba(committee, X)

    disagreement = np.var(probabilities, axis=0)

    return disagreement


def make_default_committee(random_state=42):
    """
    Create the default committee used for Query by Committee.

    Returns
    -------
    list
        List of untrained classifiers.
    """

    committee = [
        LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
        ),
        GradientBoostingClassifier(
            random_state=random_state,
        ),
    ]

    return committee


def train_committee(X_train, y_train, random_state=42):
    """
    Train every committee member.

    Returns
    -------
    list
        List of trained classifiers.
    """

    committee = make_default_committee(random_state)

    for model in committee:
        model.fit(X_train, y_train)

    return committee