from sklearn.metrics import roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on a fixed test set.

    This estimates how well the model generalizes to molecules
    that were never used for training or active-learning selection.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)

    return {
        "roc_auc": roc_auc,
    }