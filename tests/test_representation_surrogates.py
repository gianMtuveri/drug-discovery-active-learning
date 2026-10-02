from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, WhiteKernel

from src.regression.models import make_regression_surrogate
from src.regression.preprocessing import DenseVarianceStandardizer
from src.regression.representation_surrogates import (
    PreprocessedRegressionSurrogate,
    make_representation_surrogate,
    preprocessing_policy,
)


BENCHMARK_MODELS = (
    "random_forest", "extra_trees", "bayesian_ridge", "gaussian_process"
)
DENSE_REPRESENTATIONS = (
    "rdkit_descriptors", "smi_ted_light", "unimol_v1"
)


def test_v051_production_gp_matches_approved_rational_quadratic_configuration():
    surrogate = make_regression_surrogate("gaussian_process", random_state=17)
    estimator = surrogate.model
    kernel = estimator.kernel

    assert isinstance(kernel.k1.k1, ConstantKernel)
    assert isinstance(kernel.k1.k2, RationalQuadratic)
    assert isinstance(kernel.k2, WhiteKernel)
    assert kernel.k1.k1.constant_value == 1.0
    assert kernel.k1.k1.constant_value_bounds == (1e-5, 1e5)
    assert kernel.k1.k2.length_scale == 1.0
    assert kernel.k1.k2.alpha == 1.0
    assert kernel.k1.k2.length_scale_bounds == (1e-5, 1e5)
    assert kernel.k1.k2.alpha_bounds == (1e-5, 1e5)
    assert kernel.k2.noise_level == 1.0
    assert kernel.k2.noise_level_bounds == (1e-5, 1e5)
    assert estimator.alpha == 1e-6
    assert estimator.normalize_y is True
    assert estimator.optimizer == "fmin_l_bfgs_b"
    assert estimator.n_restarts_optimizer == 0
    assert estimator.random_state == 42


def _morgan_fixture():
    rng = np.random.default_rng(2025)
    X = rng.integers(0, 2, size=(36, 12), dtype=np.uint8)
    y = 1.5 * X[:, 0] - 0.7 * X[:, 3] + 0.2 * X[:, 7]
    return X[:28], y[:28], X[28:]


@pytest.mark.parametrize("model_name", BENCHMARK_MODELS)
def test_morgan_factory_preserves_legacy_predictions_uncertainty_and_configuration(model_name):
    X_train, y_train, X_test = _morgan_fixture()
    legacy = make_regression_surrogate(model_name, random_state=17)
    modular = make_representation_surrogate(model_name, "morgan", random_state=17)

    legacy.fit(X_train, y_train)
    modular.fit(X_train, y_train)

    assert type(modular) is type(legacy)
    assert modular.model.get_params(deep=True) == legacy.model.get_params(deep=True)
    legacy_prediction = legacy.predict(X_test)
    modular_prediction = modular.predict(X_test)
    legacy_mean, legacy_uncertainty = legacy.predict_with_uncertainty(X_test)
    modular_mean, modular_uncertainty = modular.predict_with_uncertainty(X_test)
    if model_name in {"random_forest", "extra_trees"}:
        np.testing.assert_array_equal(
            modular.ensemble_predictions(X_test), legacy.ensemble_predictions(X_test)
        )
        # sklearn's parallel ensemble reduction can differ at machine epsilon
        # even when every fitted tree and member prediction is identical.
        np.testing.assert_allclose(modular_prediction, legacy_prediction, rtol=0, atol=1e-15)
        np.testing.assert_allclose(modular_mean, legacy_mean, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            modular_uncertainty, legacy_uncertainty, rtol=0, atol=1e-15
        )
    else:
        np.testing.assert_allclose(modular_prediction, legacy_prediction, rtol=0, atol=1e-12)
        np.testing.assert_allclose(modular_mean, legacy_mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(modular_uncertainty, legacy_uncertainty, rtol=0, atol=1e-12)


def test_variance_filter_is_learned_from_labelled_set_only():
    labelled = np.asarray([
        [1.0, 10.0, 2.0],
        [2.0, 10.0, 4.0],
        [3.0, 10.0, 8.0],
    ])
    globally_different = np.asarray([[4.0, 999.0, 16.0]])
    preprocessor = DenseVarianceStandardizer().fit(labelled)

    assert preprocessor.retained_feature_mask_.tolist() == [True, False, True]
    assert preprocessor.transform(globally_different).shape == (1, 2)


def test_scaler_statistics_are_labelled_only_and_transform_does_not_refit():
    labelled = np.asarray([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]])
    external = np.asarray([[1000.0, -500.0], [2000.0, -1000.0]])
    preprocessor = DenseVarianceStandardizer().fit(labelled)
    mean_before = preprocessor.scaler.mean_.copy()
    scale_before = preprocessor.scaler.scale_.copy()

    transformed = preprocessor.transform(external)

    np.testing.assert_array_equal(mean_before, labelled.mean(axis=0))
    np.testing.assert_allclose(scale_before, labelled.std(axis=0, ddof=0))
    np.testing.assert_array_equal(preprocessor.scaler.mean_, mean_before)
    np.testing.assert_array_equal(preprocessor.scaler.scale_, scale_before)
    np.testing.assert_allclose(
        transformed, (external - mean_before) / scale_before, rtol=0, atol=1e-14
    )


def test_dense_preprocessor_fails_before_fit_on_dimension_change_and_all_constants():
    preprocessor = DenseVarianceStandardizer()
    with pytest.raises(RuntimeError, match="fitted before transform"):
        preprocessor.transform(np.ones((2, 3)))
    with pytest.raises(ValueError, match="All dense representation dimensions"):
        preprocessor.fit(np.ones((4, 3)))

    preprocessor.fit(np.asarray([[0.0, 1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="feature dimension differs"):
        preprocessor.transform(np.ones((2, 3)))


@pytest.mark.parametrize("representation_name", DENSE_REPRESENTATIONS)
@pytest.mark.parametrize("model_name", ("random_forest", "extra_trees"))
def test_dense_tree_models_use_raw_features(model_name, representation_name):
    model = make_representation_surrogate(model_name, representation_name, random_state=9)
    X = np.asarray([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    y = np.arange(4.0)
    model.fit(X, y)

    assert not isinstance(model, PreprocessedRegressionSurrogate)
    assert model.model.n_features_in_ == X.shape[1]
    assert preprocessing_policy(model_name, representation_name) == "raw_dense"


@pytest.mark.parametrize("representation_name", DENSE_REPRESENTATIONS)
@pytest.mark.parametrize("model_name", ("bayesian_ridge", "gaussian_process"))
def test_dense_linear_and_gp_policy_uses_labelled_variance_filter_and_scaler(
    model_name, representation_name
):
    model = make_representation_surrogate(model_name, representation_name, random_state=4)
    X_labeled = np.asarray([
        [0.0, 5.0, 10.0],
        [1.0, 5.0, 12.0],
        [2.0, 5.0, 14.0],
        [3.0, 5.0, 16.0],
    ])
    y = np.asarray([0.0, 1.0, 1.5, 3.0])
    model.fit(X_labeled, y)

    assert isinstance(model, PreprocessedRegressionSurrogate)
    assert model.preprocessor.retained_feature_mask_.tolist() == [True, False, True]
    np.testing.assert_array_equal(model.preprocessor.scaler.mean_, [1.5, 13.0])
    assert model.model.n_features_in_ == 2
    assert preprocessing_policy(model_name, representation_name) == (
        "labelled_variance_filter_then_standard_scaler"
    )


@pytest.mark.parametrize("model_name", BENCHMARK_MODELS)
def test_morgan_policy_is_legacy_for_every_model(model_name):
    assert preprocessing_policy(model_name, "morgan") == "morgan_v050"


def test_unknown_model_and_representation_fail_clearly():
    with pytest.raises(ValueError, match="Unknown benchmark surrogate"):
        make_representation_surrogate("unknown", "morgan")
    with pytest.raises(ValueError, match="Unknown molecular representation"):
        make_representation_surrogate("random_forest", "unknown")
