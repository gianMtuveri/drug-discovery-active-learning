"""Representation-aware construction of the four benchmark surrogates."""

from __future__ import annotations

import numpy as np

from src.regression.base import RegressionSurrogate
from src.regression.models import (
    BayesianRidgeSurrogate,
    GaussianProcessSurrogate,
    make_regression_surrogate,
)
from src.regression.preprocessing import DenseVarianceStandardizer


BENCHMARK_MODEL_NAMES = (
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
)
REPRESENTATION_NAMES = (
    "morgan",
    "rdkit_descriptors",
    "smi_ted_light",
    "unimol_v1",
)
DENSE_REPRESENTATIONS = frozenset(REPRESENTATION_NAMES[1:])
STANDARDIZED_DENSE_MODELS = frozenset(("bayesian_ridge", "gaussian_process"))


class PreprocessedRegressionSurrogate(RegressionSurrogate):
    """Compose one fitted dense preprocessor with an existing surrogate."""

    def __init__(
        self,
        estimator: RegressionSurrogate,
        preprocessor: DenseVarianceStandardizer,
    ) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor

    @property
    def model(self):
        """Expose the underlying sklearn estimator for inspection compatibility."""
        return self.estimator.model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreprocessedRegressionSurrogate":
        transformed = self.preprocessor.fit_transform(X)
        self.estimator.fit(transformed, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.preprocessor.transform(X))

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.estimator.predict_with_uncertainty(self.preprocessor.transform(X))

    def ensemble_predictions(self, X: np.ndarray) -> np.ndarray | None:
        return self.estimator.ensemble_predictions(self.preprocessor.transform(X))


def preprocessing_policy(model_name: str, representation_name: str) -> str:
    """Return the canonical preprocessing policy and validate identifiers."""
    if model_name not in BENCHMARK_MODEL_NAMES:
        raise ValueError(
            f"Unknown benchmark surrogate: {model_name}. "
            f"Expected one of {list(BENCHMARK_MODEL_NAMES)}"
        )
    if representation_name not in REPRESENTATION_NAMES:
        raise ValueError(
            f"Unknown molecular representation: {representation_name}. "
            f"Expected one of {list(REPRESENTATION_NAMES)}"
        )
    if representation_name == "morgan":
        return "morgan_v050"
    if model_name in STANDARDIZED_DENSE_MODELS:
        return "labelled_variance_filter_then_standard_scaler"
    return "raw_dense"


def make_representation_surrogate(
    model_name: str,
    representation_name: str,
    random_state: int | None = 42,
) -> RegressionSurrogate:
    """Construct a benchmark surrogate without changing the legacy factory."""
    policy = preprocessing_policy(model_name, representation_name)
    if policy == "morgan_v050" or policy == "raw_dense":
        return make_regression_surrogate(model_name, random_state=random_state)

    if model_name == "bayesian_ridge":
        estimator: RegressionSurrogate = BayesianRidgeSurrogate(
            use_internal_preprocessing=False
        )
    elif model_name == "gaussian_process":
        estimator = GaussianProcessSurrogate()
    else:  # Guarded by preprocessing_policy; retained as a defensive invariant.
        raise RuntimeError(f"No dense preprocessing construction for {model_name}")
    return PreprocessedRegressionSurrogate(
        estimator=estimator,
        preprocessor=DenseVarianceStandardizer(),
    )
