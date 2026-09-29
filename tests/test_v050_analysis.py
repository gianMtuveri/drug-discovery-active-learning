import json

import numpy as np
import pandas as pd
import pytest

from scripts.regression.analyze_v050_benchmark import report
from src.regression.v050_analysis import (
    add_recalculated_endpoints,
    campaign_endpoints,
    identify_source,
    load_artifacts,
    paired_comparisons,
    validate_endpoint_consistency,
)


def test_report_describes_supplied_campaign_completeness(tmp_path):
    validation = pd.DataFrame(
        [{"check": "campaign completeness", "status": "passed", "details": "ok"}]
    )
    consistency = pd.DataFrame(
        {
            "cumulative_counts_agree": [True, True],
            "auc_agrees": [True, True],
            "final_count_agrees_with_curve": [True, True],
        }
    )

    complete = pd.DataFrame({"complete_campaign": [True, True]})
    report(tmp_path, validation, complete, pd.DataFrame(), consistency)
    text = (tmp_path / "scientific_report.md").read_text()
    assert "All 2 supplied campaigns are complete" in text
    assert "Random Forest result files are absent" not in text

    incomplete = pd.DataFrame({"complete_campaign": [True, False]})
    report(tmp_path, validation, incomplete, pd.DataFrame(), consistency)
    text = (tmp_path / "scientific_report.md").read_text()
    assert "1/2 supplied campaigns are complete" in text
    assert "1 incomplete campaigns" in text


def _history(strategy="ucb", model="extra_trees", seed=0, rounds=2):
    rows = []
    batches = [([], [], []), ([0, 1], [100, 900], [.7, .8]),
               ([2, 3], [200, 300], [.5, .9])]
    cumulative_joint = 0
    cumulative_affinity = 0
    curve_x, curve_y = [], []
    for round_idx in range(rounds + 1):
        idx, affinity, qed = batches[round_idx]
        cumulative_joint += sum(a <= 500 and q >= .6 for a, q in zip(affinity, qed))
        cumulative_affinity += sum(a <= 500 for a in affinity)
        curve_x.append(round_idx * 2); curve_y.append(cumulative_joint)
        rows.append({"target": "EGFR", "strategy": strategy, "model": model,
                     "seed": seed, "round": round_idx, "n_labeled": 2 + round_idx * 2,
                     "n_initial_labeled": 2, "n_newly_acquired": 0 if round_idx == 0 else 2,
                     "n_cumulative_acquired": round_idx * 2,
                     "initial_labeled_indices": "[10, 11]",
                     "newly_acquired_indices": json.dumps(idx),
                     "cumulative_acquired_indices": json.dumps(list(range(round_idx * 2))),
                     "newly_acquired_smiles": json.dumps(["c1ccccc1"] * len(idx)),
                     "newly_acquired_affinity_nm": json.dumps(affinity),
                     "newly_acquired_qed": json.dumps(qed),
                     "joint_discovery_count": cumulative_joint,
                     "affinity_discovery_count": cumulative_affinity,
                     "joint_discovery_auc": np.trapezoid(curve_y, curve_x) if round_idx else 0,
                     "joint_scaffold_count": cumulative_joint, "joint_diversity": .5,
                     "rmse": 1., "pearson": .5})
    return pd.DataFrame(rows)


def _source(path, strategy="ucb", model="extra_trees", seed=0, rounds=2):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"target": "EGFR", "strategy": strategy,
                                                   "models": [model], "rounds": rounds}))
    _history(strategy, model, seed, rounds).to_csv(path / "history.csv", index=False)


def test_strategy_identification_requires_config_history_agreement(tmp_path):
    _source(tmp_path / "source")
    frame = pd.read_csv(tmp_path / "source" / "history.csv")
    frame["strategy"] = "ucb_qed"
    frame.to_csv(tmp_path / "source" / "history.csv", index=False)
    with pytest.raises(ValueError, match="Strategy conflict"):
        identify_source(tmp_path / "source")


def test_duplicate_sources_choose_complete_and_require_matching_provenance(tmp_path):
    _source(tmp_path / "short", rounds=1)
    _source(tmp_path / "complete", rounds=2)
    loaded = load_artifacts(tmp_path)
    assert loaded.history["round"].tolist() == [0, 1, 2]
    assert len(loaded.overlaps) == 1
    assert bool(loaded.overlaps.iloc[0].molecular_provenance_agrees)
    frame = pd.read_csv(tmp_path / "short" / "history.csv")
    frame.loc[frame["round"] == 1, "newly_acquired_indices"] = "[98, 99]"
    frame.to_csv(tmp_path / "short" / "history.csv", index=False)
    with pytest.raises(ValueError, match="Conflicting molecular provenance"):
        load_artifacts(tmp_path)


def test_endpoint_calculation_excludes_initial_and_matches_curve():
    history = add_recalculated_endpoints(_history())
    consistency = validate_endpoint_consistency(history)
    assert consistency.iloc[0][["cumulative_counts_agree", "auc_agrees",
                                "final_count_agrees_with_curve"]].all()
    endpoint = campaign_endpoints(history, expected_rounds=2).iloc[0]
    assert endpoint.joint_discovery_count == 2
    assert endpoint.affinity_discovery_count == 3
    assert endpoint.qed_fraction_of_affinity_discoveries == pytest.approx(2 / 3)


def test_paired_differences_and_interaction_are_seed_matched():
    rows = []
    values = {"ucb": 10, "ucb_qed": 13, "ucb_diversity": 12,
              "ucb_qed_diversity": 18}
    for strategy, value in values.items():
        row = _history(strategy=strategy, rounds=0).iloc[0].to_dict()
        row.update({"joint_discovery_count": value, "affinity_discovery_count": value + 1,
                    "joint_discovery_auc": value * 10, "joint_scaffold_count": value,
                    "joint_diversity": value / 20, "rmse": 1, "pearson": .5})
        rows.append(row)
    endpoints = campaign_endpoints(pd.DataFrame(rows), expected_rounds=0)
    seed, summary = paired_comparisons(endpoints, metrics=["joint_discovery_count"])
    got = seed.set_index("comparison").paired_difference.to_dict()
    assert got["adding_qed_without_diversity"] == 3
    assert got["adding_qed_with_diversity"] == 6
    assert got["adding_diversity_without_qed"] == 2
    assert got["adding_diversity_with_qed"] == 5
    assert got["qed_by_diversity_interaction"] == 3
    assert summary.n_seeds.eq(1).all()
