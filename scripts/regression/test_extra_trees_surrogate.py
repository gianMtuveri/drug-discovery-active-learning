import numpy as np
from sklearn.datasets import make_regression

from src.regression.models import make_regression_surrogate


def test_extra_trees_surrogate():

    X, y = make_regression(
        n_samples=100,
        n_features=20,
        noise=0.1,
        random_state=42,
    )

    model = make_regression_surrogate(
        model_name="extra_trees",
        random_state=42,
    )

    returned = model.fit(X, y)

    # fit returns self
    assert returned is model

    prediction = model.predict(X)

    mean, uncertainty = (
        model.predict_with_uncertainty(X)
    )

    ensemble = model.ensemble_predictions(X)

    # interface
    assert prediction.shape == (100,)
    assert mean.shape == (100,)
    assert uncertainty.shape == (100,)

    # ensemble dimensions
    assert ensemble.shape[1] == 100

    # prediction consistency
    np.testing.assert_allclose(
        prediction,
        mean,
        atol=1e-12,
    )

    # uncertainty must be non-negative
    assert np.all(uncertainty >= 0)

    # ensemble mean == predict()
    np.testing.assert_allclose(
        ensemble.mean(axis=0),
        prediction,
        atol=1e-12,
    )