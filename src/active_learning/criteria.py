from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcquisitionContext:
    """
    Data available to acquisition criteria for the current pool.

    All pool-level arrays must follow the same ordering as
    unlabeled_indices.
    """

    unlabeled_indices: np.ndarray
    predicted_mean: np.ndarray | None = None
    predicted_uncertainty: np.ndarray | None = None
    X_pool: np.ndarray | None = None

    def __post_init__(self) -> None:
        unlabeled_indices = np.asarray(
            self.unlabeled_indices
        )

        if unlabeled_indices.ndim != 1:
            raise ValueError(
                "unlabeled_indices must be one-dimensional."
            )

        n_pool = len(unlabeled_indices)

        arrays_to_check = {
            "predicted_mean": self.predicted_mean,
            "predicted_uncertainty": (
                self.predicted_uncertainty
            ),
        }

        for name, values in arrays_to_check.items():
            if values is None:
                continue

            values = np.asarray(values)

            if values.ndim != 1:
                raise ValueError(
                    f"{name} must be one-dimensional."
                )

            if len(values) != n_pool:
                raise ValueError(
                    f"{name} has length {len(values)}, "
                    f"but the unlabeled pool has length "
                    f"{n_pool}."
                )

        if self.X_pool is not None:
            X_pool = np.asarray(self.X_pool)

            if X_pool.shape[0] != n_pool:
                raise ValueError(
                    "X_pool and unlabeled_indices must "
                    "contain the same number of molecules."
                )


class AcquisitionCriterion(ABC):
    """
    Base class for one acquisition criterion.

    Subclasses only calculate raw scores. Direction handling,
    normalization, and weighting are managed by the engine.
    """

    VALID_DIRECTIONS = {
        "maximize",
        "minimize",
    }

    VALID_NORMALIZATIONS = {
        "none",
        "robust",
    }

    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        direction: str = "maximize",
        normalization: str = "robust",
    ) -> None:
        if not name:
            raise ValueError(
                "Criterion name cannot be empty."
            )

        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Expected one of "
                f"{sorted(self.VALID_DIRECTIONS)}."
            )

        if normalization not in (
            self.VALID_NORMALIZATIONS
        ):
            raise ValueError(
                f"Invalid normalization "
                f"'{normalization}'. Expected one of "
                f"{sorted(self.VALID_NORMALIZATIONS)}."
            )

        if not np.isfinite(weight):
            raise ValueError(
                "Criterion weight must be finite."
            )

        self.name = name
        self.weight = float(weight)
        self.direction = direction
        self.normalization = normalization

    @abstractmethod
    def raw_score(
        self,
        context: AcquisitionContext,
    ) -> np.ndarray:
        """
        Return one raw score per unlabeled molecule.
        """


class PredictionCriterion(AcquisitionCriterion):
    def __init__(
        self,
        *,
        weight: float = 1.0,
        normalization: str = "robust",
        direction: str = "maximize",
    ) -> None:
        super().__init__(
            name="prediction",
            weight=weight,
            direction=direction,
            normalization=normalization,
        )

    def raw_score(
        self,
        context: AcquisitionContext,
    ) -> np.ndarray:
        if context.predicted_mean is None:
            raise ValueError(
                "PredictionCriterion requires "
                "context.predicted_mean."
            )

        return np.asarray(
            context.predicted_mean,
            dtype=float,
        )


class UncertaintyCriterion(AcquisitionCriterion):
    def __init__(
        self,
        *,
        weight: float = 1.0,
        normalization: str = "robust",
        direction: str = "maximize",
    ) -> None:
        super().__init__(
            name="uncertainty",
            weight=weight,
            direction=direction,
            normalization=normalization,
        )

    def raw_score(
        self,
        context: AcquisitionContext,
    ) -> np.ndarray:
        if context.predicted_uncertainty is None:
            raise ValueError(
                "UncertaintyCriterion requires "
                "context.predicted_uncertainty."
            )

        return np.asarray(
            context.predicted_uncertainty,
            dtype=float,
        )