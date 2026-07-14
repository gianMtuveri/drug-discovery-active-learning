from src.active_learning.acquisition import (
    WeightedAcquisitionEngine,
)
from src.active_learning.criteria import (
    PredictionCriterion,
    UncertaintyCriterion,
)


def make_prediction_engine(
    *,
    normalization: str = "robust",
) -> WeightedAcquisitionEngine:
    return WeightedAcquisitionEngine(
        criteria=[
            PredictionCriterion(
                weight=1.0,
                normalization=normalization,
            )
        ]
    )


def make_uncertainty_engine(
    *,
    normalization: str = "robust",
) -> WeightedAcquisitionEngine:
    return WeightedAcquisitionEngine(
        criteria=[
            UncertaintyCriterion(
                weight=1.0,
                normalization=normalization,
            )
        ]
    )


def make_ucb_engine(
    *,
    beta: float,
    normalization: str = "robust",
) -> WeightedAcquisitionEngine:
    if beta < 0:
        raise ValueError(
            "beta must be non-negative."
        )

    return WeightedAcquisitionEngine(
        criteria=[
            PredictionCriterion(
                weight=1.0,
                normalization=normalization,
            ),
            UncertaintyCriterion(
                weight=beta,
                normalization=normalization,
            ),
        ]
    )