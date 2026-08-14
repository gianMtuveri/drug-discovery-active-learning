from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.regression.models import (
    make_regression_surrogate,
)


MODEL_NAMES = [
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
    "gradient_boosting",
    "hist_gradient_boosting",
    "knn",
    "linear_regression",
]


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_regression_surrogate_interface(
    model_name: str,
) -> None:
    X, y = make_regression(
        n_samples=120,
        n_features=20,
        n_informative=10,
        noise=0.5,
        random_state=42,
    )

    X_train, X_test, y_train, _ = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )
    )

    model = make_regression_surrogate(
        model_name=model_name,
        random_state=42,
    )

    returned_model = model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mean_prediction, uncertainty = (
        model.predict_with_uncertainty(
            X_test
        )
    )

    assert returned_model is model

    assert predictions.shape == (
        len(X_test),
    )

    assert mean_prediction.shape == (
        len(X_test),
    )

    assert uncertainty.shape == (
        len(X_test),
    )

    assert np.all(
        np.isfinite(predictions)
    )

    assert np.all(
        np.isfinite(mean_prediction)
    )

    assert np.all(
        np.isfinite(uncertainty)
    )

    assert np.all(
        uncertainty >= 0.0
    )

    assert np.allclose(
        predictions,
        mean_prediction,
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "random_forest",
        "extra_trees",
    ],
)
def test_tree_surrogate_exposes_ensemble_predictions(
    model_name: str,
) -> None:
    X, y = make_regression(
        n_samples=80,
        n_features=10,
        random_state=42,
    )

    model = make_regression_surrogate(
        model_name=model_name,
        random_state=42,
    )

    model.fit(X, y)

    ensemble_predictions = (
        model.ensemble_predictions(
            X[:5]
        )
    )

    assert ensemble_predictions is not None

    assert ensemble_predictions.shape[1] == 5

    assert ensemble_predictions.shape[0] > 1


def test_bayesian_ridge_has_no_ensemble_predictions() -> None:
    X, y = make_regression(
        n_samples=80,
        n_features=10,
        random_state=42,
    )

    model = make_regression_surrogate(
        model_name="bayesian_ridge",
    )

    model.fit(X, y)

    assert model.ensemble_predictions(
        X[:5]
    ) is None


def test_gaussian_process_has_no_ensemble_predictions() -> None:
    X, y = make_regression(
        n_samples=80,
        n_features=10,
        random_state=42,
    )

    model = make_regression_surrogate(
        "gaussian_process"
    )

    model.fit(X, y)

    assert model.ensemble_predictions(
        X[:5]
    ) is None


@pytest.mark.parametrize(
    "model_name",
    [
        "gradient_boosting",
        "hist_gradient_boosting",
        "knn",
        "linear_regression",
    ],
)
def test_deterministic_models_returns_zero_uncertainty(
    model_name: str,
):
    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(40, 8),
    )
    y = rng.normal(
        size=40,
    )

    model = make_regression_surrogate(
        model_name,
    )

    model.fit(X, y)

    mean, uncertainty = model.predict_with_uncertainty(
        X[:5],
    )

    assert mean.shape == (5,)
    assert uncertainty.shape == (5,)
    assert np.all(uncertainty == 0.0)
    assert model.ensemble_predictions(X[:5]) is None


def test_unknown_surrogate_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown regression surrogate",
    ):
        make_regression_surrogate(
            model_name="unknown_model",
        )