from dataclasses import dataclass

import numpy as np

from src.active_learning.acquisition import (
    WeightedAcquisitionResult,
)


@dataclass(frozen=True)
class BatchSelectionResult:
    selected_indices: np.ndarray
    selected_pool_positions: np.ndarray
    selected_combined_scores: np.ndarray


class TopKBatchSelector:
    """
    Select the highest-scoring molecules from the pool.
    """

    def select(
        self,
        acquisition_result: WeightedAcquisitionResult,
        batch_size: int,
    ) -> BatchSelectionResult:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        pool_size = len(
            acquisition_result.unlabeled_indices
        )

        n_select = min(
            batch_size,
            pool_size,
        )

        # Stable sorting preserves deterministic behavior
        # for tied scores.
        order = np.argsort(
            acquisition_result.combined_scores,
            kind="stable",
        )[::-1]

        selected_pool_positions = order[
            :n_select
        ]

        selected_indices = (
            acquisition_result.unlabeled_indices[
                selected_pool_positions
            ]
        )

        selected_scores = (
            acquisition_result.combined_scores[
                selected_pool_positions
            ]
        )

        return BatchSelectionResult(
            selected_indices=selected_indices,
            selected_pool_positions=(
                selected_pool_positions
            ),
            selected_combined_scores=(
                selected_scores
            ),
        )