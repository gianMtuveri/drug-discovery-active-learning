from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.regression.run_v051_representation_benchmark import (
    CampaignSpec,
    EXPECTED_CAMPAIGNS,
    campaign_matrix,
    validate_campaign_history,
)


def test_frozen_matrix_contains_960_campaigns_without_extra_trees():
    matrix = campaign_matrix()
    assert len(matrix) == EXPECTED_CAMPAIGNS == 960
    assert len(set(matrix)) == 960
    assert {spec.seed for spec in matrix} == set(range(20))
    assert {spec.model for spec in matrix} == {
        "random_forest", "bayesian_ridge", "gaussian_process"
    }
    assert all(spec.model != "extra_trees" for spec in matrix)


def _complete_history(spec: CampaignSpec) -> pd.DataFrame:
    frame = pd.DataFrame({
        "round": range(21),
        "representation": spec.representation,
        "model": spec.model,
        "strategy": spec.acquisition,
        "seed": spec.seed,
        "population_hash": "hash",
        "preprocessing_policy": "raw_dense",
        "model_feature_dimension": 8,
        "morgan_feature_dimension": 2048,
        "beta": 2.0,
        "n_newly_acquired": [0] + [10] * 20,
        "n_cumulative_acquired": range(0, 201, 10),
        "newly_acquired_indices": [json.dumps([])] + [
            json.dumps(list(range(i, i + 10))) for i in range(0, 200, 10)
        ],
        "initial_labeled_indices": json.dumps(list(range(300, 320))),
    })
    for column in (
        "rmse", "mae", "r2", "pearson", "best_discovered",
        "top20_mean_discovered", "mean_discovered", "dataset_best",
        "fraction_best_found", "distance_to_best", "round_runtime_seconds",
        "affinity_discovery_count", "joint_discovery_count",
        "joint_discovery_auc", "joint_scaffold_count",
    ):
        frame[column] = 1.0
    for column in ("pool_mean_uncertainty", "pool_max_uncertainty"):
        frame[column] = [1.0] * 20 + [np.nan]
    for column in (
        "selected_mean_prediction", "selected_mean_uncertainty",
        "selected_mean_combined_score", "selected_min_combined_score",
        "selected_max_combined_score",
    ):
        frame[column] = [np.nan] + [1.0] * 20
    return frame


def test_completed_campaign_validation_rejects_partial_and_test_acquisition():
    spec = CampaignSpec("rdkit_descriptors", "random_forest", "ucb", 0)
    frame = _complete_history(spec)
    result = validate_campaign_history(
        frame, spec, population_hash="hash", held_out_test_indices=(500,),
        initial_labeled_indices=tuple(range(300, 320)),
    )
    assert result["acquired_count"] == 200
    with pytest.raises(ValueError, match="rounds 0..20"):
        validate_campaign_history(
            frame.iloc[:-1], spec, population_hash="hash",
            held_out_test_indices=(500,),
        )
    with pytest.raises(ValueError, match="held-out"):
        validate_campaign_history(
            frame, spec, population_hash="hash", held_out_test_indices=(50,),
        )


def test_validation_preserves_undefined_constant_input_pearson_but_rejects_inf():
    spec = CampaignSpec("rdkit_descriptors", "random_forest", "ucb", 0)
    frame = _complete_history(spec)
    frame.loc[1, "pearson"] = np.nan
    result = validate_campaign_history(
        frame, spec, population_hash="hash", held_out_test_indices=(500,)
    )
    assert result["undefined_pearson_rounds"] == 1
    frame.loc[1, "pearson"] = np.inf
    with pytest.raises(ValueError, match="Infinite campaign output: pearson"):
        validate_campaign_history(
            frame, spec, population_hash="hash", held_out_test_indices=(500,)
        )
