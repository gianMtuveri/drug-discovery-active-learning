"""Labelled-set preprocessing for dense molecular representations."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class DenseVarianceStandardizer:
    """Remove labelled-set constants, then standardize retained dimensions."""

    def __init__(self) -> None:
        self.retained_feature_mask_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.scaler: StandardScaler | None = None

    @property
    def is_fitted(self) -> bool:
        return self.retained_feature_mask_ is not None and self.scaler is not None

    def fit(self, X_train: np.ndarray) -> "DenseVarianceStandardizer":
        X = self._validate_2d_finite(X_train, "X_train")
        self.n_features_in_ = X.shape[1]
        variances = np.var(X, axis=0)
        retained = variances > 0.0
        if not retained.any():
            raise ValueError(
                "All dense representation dimensions are constant in the labelled training set"
            )
        self.retained_feature_mask_ = retained
        self.scaler = StandardScaler()
        self.scaler.fit(X[:, retained])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DenseVarianceStandardizer must be fitted before transform")
        values = self._validate_2d_finite(X, "X")
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                "Dense representation feature dimension differs from fitted preprocessing: "
                f"{values.shape[1]} != {self.n_features_in_}"
            )
        return self.scaler.transform(values[:, self.retained_feature_mask_])

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @staticmethod
    def _validate_2d_finite(X: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional feature matrix")
        if values.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{name} must be numeric")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        return values
