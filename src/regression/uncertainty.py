import numpy as np
from sklearn.ensemble import RandomForestRegressor


def predict_with_uncertainty(
    model: RandomForestRegressor,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate predictive mean and uncertainty from a Random Forest.

    The mean is the average prediction across trees.
    The uncertainty is the standard deviation across tree predictions.
    """
    if not hasattr(model, "estimators_"):
        raise ValueError("The Random Forest model must be fitted first.")

    tree_predictions = np.vstack(
        [tree.predict(X) for tree in model.estimators_]
    )

    mean_prediction = tree_predictions.mean(axis=0)
    uncertainty = tree_predictions.std(axis=0)

    return mean_prediction, uncertainty