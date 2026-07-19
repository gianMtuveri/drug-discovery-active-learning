from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class RegressionSurrogate(ABC):
    """Common interface for regression surrogate models."""

    @abstractmethod
    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
    ) -> "RegressionSurrogate":
        """Fit the surrogate model."""

    @abstractmethod
    def predict(
        self,
        X: FloatArray,
    ) -> FloatArray:
        """Return point predictions."""

    @abstractmethod
    def predict_with_uncertainty(
        self,
        X: FloatArray,
    ) -> Tuple[FloatArray, FloatArray]:
        """Return predictive mean and uncertainty."""

    def ensemble_predictions(
        self,
        X: FloatArray,
    ) -> Optional[FloatArray]:
        """Return member-level predictions when available."""
        return None