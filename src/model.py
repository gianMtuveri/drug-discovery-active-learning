from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_model(X, y, labeled_indices):
    """
    Train a classifier using only experimentally tested molecules.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for all molecules.
    y : np.ndarray
        True labels for all molecules.
        Only labels at labeled_indices are used.
    labeled_indices : np.ndarray
        Indices of molecules already tested experimentally.

    Returns
    -------
    model : sklearn Pipeline
        Trained classification model.
    """

    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )

    model.fit(X_labeled, y_labeled)

    return model