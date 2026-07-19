from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from src.regression.base import RegressionSurrogate

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)


FloatArray = NDArray[np.float64]


def default_gaussian_process_kernel():
    """
    Default kernel used by GaussianProcessSurrogate.

    The kernel can be replaced in the future once benchmarking
    identifies more suitable choices for molecular fingerprints.
    """
    return (
        ConstantKernel(1.0)
        * RBF(length_scale=1.0)
        + WhiteKernel()
    )


class TreeEnsembleSurrogate(
    RegressionSurrogate,
    ABC,
):
    """Shared implementation for tree-based ensembles."""

    model: Any

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
    ) -> "TreeEnsembleSurrogate":
        self.model.fit(X, y)
        return self

    def predict(
        self,
        X: FloatArray,
    ) -> FloatArray:
        return np.asarray(
            self.model.predict(X),
            dtype=float,
        )

    def ensemble_predictions(
        self,
        X: FloatArray,
    ) -> FloatArray:
        return np.vstack(
            [
                estimator.predict(X)
                for estimator in self.model.estimators_
            ]
        )

    def predict_with_uncertainty(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        predictions = self.ensemble_predictions(X)

        mean_prediction = predictions.mean(
            axis=0
        )

        uncertainty = predictions.std(
            axis=0
        )

        return (
            np.asarray(
                mean_prediction,
                dtype=float,
            ),
            np.asarray(
                uncertainty,
                dtype=float,
            ),
        )


class RandomForestSurrogate(
    TreeEnsembleSurrogate,
):
    """Random Forest regression surrogate."""

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int | None = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )


class ExtraTreesSurrogate(
    TreeEnsembleSurrogate,
):
    """Extra Trees regression surrogate."""

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int | None = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )


class BayesianRidgeSurrogate(
    RegressionSurrogate,
):
    """
    Bayesian linear regression surrogate.

    Predictive uncertainty is obtained directly from the
    posterior predictive distribution estimated by
    sklearn.linear_model.BayesianRidge.
    """

    def __init__(
        self,
        *,
        max_iter: int = 300,
        tol: float = 1e-3,
        fit_intercept: bool = True,
    ) -> None:
        # with_mean=False avoids densification if sparse fingerprint
        # matrices are introduced later.
        self.scaler = StandardScaler(
            with_mean=False,
        )

        self.model = BayesianRidge(
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
        )

        self._is_fitted = False

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
    ) -> "BayesianRidgeSurrogate":
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(
            X_scaled,
            y,
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        X: FloatArray,
    ) -> FloatArray:
        self._check_is_fitted()

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(
            X_scaled
        )

        return np.asarray(
            predictions,
            dtype=float,
        )

    def predict_with_uncertainty(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        self._check_is_fitted()

        X_scaled = self.scaler.transform(X)

        mean_prediction, uncertainty = (
            self.model.predict(
                X_scaled,
                return_std=True,
            )
        )

        # Numerical noise should not normally produce negative
        # values, but clipping protects the common interface.
        uncertainty = np.maximum(
            uncertainty,
            0.0,
        )

        return (
            np.asarray(
                mean_prediction,
                dtype=float,
            ),
            np.asarray(
                uncertainty,
                dtype=float,
            ),
        )

    def _check_is_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "BayesianRidgeSurrogate must be fitted "
                "before prediction."
            )
        

class GaussianProcessSurrogate(
    RegressionSurrogate,
):
    """
    Gaussian Process regression surrogate.

    Predictive uncertainty is obtained directly from the posterior
    predictive distribution estimated by Gaussian Process Regression.
    """

    def __init__(
        self,
        *,
        alpha: float = 1e-6,
        normalize_y: bool = True,
    ) -> None:

        self.model = GaussianProcessRegressor(
            kernel=default_gaussian_process_kernel(),
            alpha=alpha,
            normalize_y=normalize_y,
            random_state=42,
        )

        self._is_fitted = False

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
    ) -> "GaussianProcessSurrogate":

        self.model.fit(
            X,
            y,
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        X: FloatArray,
    ) -> FloatArray:

        self._check_is_fitted()

        predictions = self.model.predict(X)

        return np.asarray(
            predictions,
            dtype=float,
        )

    def predict_with_uncertainty(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:

        self._check_is_fitted()

        mean_prediction, uncertainty = (
            self.model.predict(
                X,
                return_std=True,
            )
        )

        uncertainty = np.maximum(
            uncertainty,
            0.0,
        )

        return (
            np.asarray(
                mean_prediction,
                dtype=float,
            ),
            np.asarray(
                uncertainty,
                dtype=float,
            ),
        )

    def _check_is_fitted(
        self,
    ) -> None:

        if not self._is_fitted:
            raise RuntimeError(
                "GaussianProcessSurrogate must be fitted "
                "before prediction."
            )


class DeterministicRegressionSurrogate(
    RegressionSurrogate,
    ABC,
):
    """
    Shared implementation for deterministic regression surrogates.

    These estimators provide point predictions but no native
    predictive uncertainty. The uncertainty interface therefore
    returns a zero vector explicitly.
    """

    model: Any
    _is_fitted: bool

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
    ) -> "DeterministicRegressionSurrogate":
        self.model.fit(X, y)

        self._is_fitted = True

        return self

    def predict(
        self,
        X: FloatArray,
    ) -> FloatArray:
        self._check_is_fitted()

        predictions = self.model.predict(X)

        return np.asarray(
            predictions,
            dtype=float,
        )

    def predict_with_uncertainty(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        predictions = self.predict(X)

        uncertainty = np.zeros_like(
            predictions,
            dtype=float,
        )

        return predictions, uncertainty

    def _check_is_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted "
                "before prediction."
            )


class GradientBoostingSurrogate(
    DeterministicRegressionSurrogate,
):
    """Gradient Boosting regression surrogate."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )

        self._is_fitted = False


class HistGradientBoostingSurrogate(
    DeterministicRegressionSurrogate,
):
    """Histogram-based Gradient Boosting regression surrogate."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.0,
        early_stopping: bool | str = "auto",
        random_state: int | None = 42,
    ) -> None:
        self.model = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            random_state=random_state,
        )

        self._is_fitted = False


class KNearestNeighborsSurrogate(
    DeterministicRegressionSurrogate,
):
    """k-Nearest Neighbors regression surrogate."""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int = -1,
    ) -> None:

        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=n_jobs,
        )

        self._is_fitted = False


class LinearRegressionSurrogate(
    DeterministicRegressionSurrogate,
):
    """Ordinary Least Squares regression surrogate."""

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            n_jobs=n_jobs,
        )

        self._is_fitted = False


def make_regression_surrogate(
    model_name: str = "random_forest",
    random_state: int | None = 42,
) -> RegressionSurrogate:
    """
    Construct a regression surrogate by name.

    Parameters
    ----------
    model_name
        Registered surrogate name.
    random_state
        Random seed used by stochastic estimators. Bayesian Ridge
        is deterministic and therefore does not use this argument.
    """
    if model_name == "random_forest":
        return RandomForestSurrogate(
            random_state=random_state,
        )

    if model_name == "extra_trees":
        return ExtraTreesSurrogate(
            random_state=random_state,
        )

    if model_name == "bayesian_ridge":
        return BayesianRidgeSurrogate()
    
    if model_name == "gaussian_process":
        return GaussianProcessSurrogate()
    
    if model_name == "gradient_boosting":
        return GradientBoostingSurrogate(
            random_state=random_state,
        )
    
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingSurrogate(
            random_state=random_state,
        )
    if model_name == "knn":
        return KNearestNeighborsSurrogate()
    
    if model_name == "linear_regression":
        return LinearRegressionSurrogate()
    
    available_models = [
        "random_forest",
        "extra_trees",
        "bayesian_ridge",
        "gaussian_process",
        "gradient_boosting",
        "hist_gradient_boosting",
        "knn",
        "linear_regression",
    ]

    raise ValueError(
        f"Unknown regression surrogate: {model_name}. "
        f"Available models: {available_models}"
    )