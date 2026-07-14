from dataclasses import dataclass

import numpy as np

from src.active_learning.criteria import (
    AcquisitionContext,
    AcquisitionCriterion,
)
from src.active_learning.normalization import (
    normalize_scores,
)


@dataclass(frozen=True)
class CriterionEvaluation:
    name: str
    weight: float
    direction: str
    normalization: str
    raw_scores: np.ndarray
    directed_scores: np.ndarray
    normalized_scores: np.ndarray
    weighted_contribution: np.ndarray


@dataclass(frozen=True)
class WeightedAcquisitionResult:
    """
    Full output of a weighted acquisition calculation.
    """

    unlabeled_indices: np.ndarray
    combined_scores: np.ndarray
    criterion_evaluations: dict[
        str,
        CriterionEvaluation,
    ]

    def get_evaluation(
        self,
        criterion_name: str,
    ) -> CriterionEvaluation:
        try:
            return self.criterion_evaluations[
                criterion_name
            ]
        except KeyError as error:
            available = ", ".join(
                sorted(self.criterion_evaluations)
            )

            raise KeyError(
                f"Criterion '{criterion_name}' was not "
                f"evaluated. Available criteria: "
                f"{available}"
            ) from error


class WeightedAcquisitionEngine:
    """
    Combine normalized acquisition criteria using a weighted sum.

    Every criterion is converted to a convention where larger
    values are preferable before normalization and weighting.
    """

    def __init__(
        self,
        criteria: list[AcquisitionCriterion],
    ) -> None:
        if not criteria:
            raise ValueError(
                "At least one acquisition criterion "
                "is required."
            )

        criterion_names = [
            criterion.name
            for criterion in criteria
        ]

        duplicate_names = {
            name
            for name in criterion_names
            if criterion_names.count(name) > 1
        }

        if duplicate_names:
            raise ValueError(
                "Criterion names must be unique. "
                f"Duplicates: "
                f"{sorted(duplicate_names)}"
            )

        self.criteria = list(criteria)

    def evaluate(
        self,
        context: AcquisitionContext,
    ) -> WeightedAcquisitionResult:
        n_pool = len(context.unlabeled_indices)

        combined_scores = np.zeros(
            n_pool,
            dtype=float,
        )

        evaluations: dict[
            str,
            CriterionEvaluation,
        ] = {}

        for criterion in self.criteria:
            raw_scores = np.asarray(
                criterion.raw_score(context),
                dtype=float,
            )

            self._validate_scores(
                criterion=criterion,
                raw_scores=raw_scores,
                expected_length=n_pool,
            )

            if criterion.direction == "maximize":
                directed_scores = raw_scores.copy()
            else:
                directed_scores = -raw_scores

            normalized_scores = normalize_scores(
                directed_scores,
                method=criterion.normalization,
            )

            weighted_contribution = (
                criterion.weight
                * normalized_scores
            )

            combined_scores += weighted_contribution

            evaluations[criterion.name] = (
                CriterionEvaluation(
                    name=criterion.name,
                    weight=criterion.weight,
                    direction=criterion.direction,
                    normalization=(
                        criterion.normalization
                    ),
                    raw_scores=raw_scores,
                    directed_scores=directed_scores,
                    normalized_scores=(
                        normalized_scores
                    ),
                    weighted_contribution=(
                        weighted_contribution
                    ),
                )
            )

        return WeightedAcquisitionResult(
            unlabeled_indices=np.asarray(
                context.unlabeled_indices
            ).copy(),
            combined_scores=combined_scores,
            criterion_evaluations=evaluations,
        )

    @staticmethod
    def _validate_scores(
        *,
        criterion: AcquisitionCriterion,
        raw_scores: np.ndarray,
        expected_length: int,
    ) -> None:
        if raw_scores.ndim != 1:
            raise ValueError(
                f"Criterion '{criterion.name}' returned "
                "a non-one-dimensional score array."
            )

        if len(raw_scores) != expected_length:
            raise ValueError(
                f"Criterion '{criterion.name}' returned "
                f"{len(raw_scores)} scores for a pool "
                f"of size {expected_length}."
            )

        if not np.isfinite(raw_scores).all():
            raise ValueError(
                f"Criterion '{criterion.name}' returned "
                "non-finite scores."
            )