import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.regression.models import (
    make_regression_surrogate,
)
from src.regression.uncertainty import (
    predict_with_uncertainty,
)


def test_random_forest_surrogate_equivalence() -> None:
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 12))
    y = (
        2.0 * X[:, 0]
        - 0.8 * X[:, 1]
        + rng.normal(scale=0.1, size=100)
    )

    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]

    legacy_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    legacy_model.fit(X_train, y_train)

    surrogate = make_regression_surrogate(
        model_name="random_forest",
        random_state=42,
    )
    surrogate.fit(X_train, y_train)

    legacy_prediction = legacy_model.predict(X_test)
    surrogate_prediction = surrogate.predict(X_test)

    legacy_mean, legacy_uncertainty = (
        predict_with_uncertainty(
            legacy_model,
            X_test,
        )
    )

    surrogate_mean, surrogate_uncertainty = (
        surrogate.predict_with_uncertainty(
            X_test
        )
    )

    np.testing.assert_allclose(
        surrogate_prediction,
        legacy_prediction,
    )

    np.testing.assert_allclose(
        surrogate_mean,
        legacy_mean,
    )

    np.testing.assert_allclose(
        surrogate_uncertainty,
        legacy_uncertainty,
    )