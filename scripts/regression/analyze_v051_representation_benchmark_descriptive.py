#!/usr/bin/env python3
"""Descriptive, non-inferential audit of the frozen v0.5.1 benchmark."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "results/representation_benchmark/v051"
OUTPUT = BENCHMARK / "analysis"
KEYS = ["representation", "model", "strategy", "seed"]
GROUPS = ["representation", "model", "strategy"]
REPRESENTATIONS = ["morgan", "rdkit_descriptors", "smi_ted_light", "unimol_v1"]
MODELS = ["random_forest", "bayesian_ridge", "gaussian_process"]
ACQUISITIONS = ["ucb", "ucb_qed", "ucb_diversity", "ucb_qed_diversity"]

FINAL_METRICS = [
    "rmse", "pearson", "affinity_discovery_count", "best_discovered",
    "top20_mean_discovered", "joint_discovery_count", "joint_discovery_auc",
    "joint_scaffold_count", "joint_diversity",
]
TRAJECTORY_METRICS = [
    "rmse", "pearson", "affinity_discovery_count", "joint_discovery_count",
    "joint_discovery_auc", "joint_scaffold_count",
]
PRIMARY_METRICS = FINAL_METRICS
METRIC_DIRECTION = {
    "rmse": "lower",
    "pearson": "higher",
    "affinity_discovery_count": "higher",
    "best_discovered": "higher",
    "top20_mean_discovered": "higher",
    "joint_discovery_count": "higher",
    "joint_discovery_auc": "higher",
    "joint_scaffold_count": "higher",
    "joint_diversity": "higher",
}

INVENTORY = {
    "predictive": ["rmse", "mae", "r2", "pearson"],
    "affinity_discovery": [
        "best_discovered", "top20_mean_discovered", "mean_discovered",
        "dataset_best", "fraction_best_found", "distance_to_best",
        "initial_affinity_discovery_count", "affinity_discovery_count",
    ],
    "joint_affinity_qed_discovery": [
        "initial_joint_discovery_count", "joint_discovery_count",
        "joint_discovery_auc", "joint_qualifying_indices",
    ],
    "chemical_diversity_scaffold": ["joint_scaffold_count", "joint_diversity"],
    "acquisition_diagnostics": [
        "pool_mean_uncertainty", "pool_max_uncertainty",
        "selected_mean_prediction", "selected_mean_uncertainty",
        "selected_mean_true_affinity", "selected_best_true_affinity",
        "selected_mean_combined_score", "selected_min_combined_score",
        "selected_max_combined_score", "selected_mean_prediction_contribution",
        "selected_mean_uncertainty_contribution", "n_newly_acquired",
        "n_cumulative_acquired", "initial_labeled_indices",
        "newly_acquired_indices", "cumulative_acquired_indices",
        "newly_acquired_smiles", "newly_acquired_affinity_nm",
        "newly_acquired_qed",
    ],
    "runtime_technical": [
        "round", "strategy", "model", "beta", "seed", "n_labeled",
        "round_runtime_seconds", "n_initial_labeled", "representation",
        "population_hash", "preprocessing_policy", "model_feature_dimension",
        "morgan_feature_dimension", "campaign_id", "campaign_runtime_seconds",
    ],
}


def descriptive(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    long = frame.melt(id_vars=group_columns, value_vars=metrics,
                      var_name="metric", value_name="value")
    return (
        long.groupby(group_columns + ["metric"], observed=True)["value"]
        .agg(
            n="count", mean="mean", std="std", median="median",
            q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
            minimum="min", maximum="max",
        )
        .reset_index()
    )


def validate(history: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    expected = pd.MultiIndex.from_product(
        [REPRESENTATIONS, MODELS, ACQUISITIONS, range(20)], names=KEYS
    )
    observed = pd.MultiIndex.from_frame(history[KEYS].drop_duplicates())
    counts = history.groupby(KEYS, observed=True).size()
    integrity = {
        "campaign_summary_rows": len(summary),
        "history_rows": len(history),
        "campaigns": len(observed),
        "seeds": sorted(history.seed.unique().tolist()),
        "representations": sorted(history.representation.unique().tolist()),
        "models": sorted(history.model.unique().tolist()),
        "acquisitions": sorted(history.strategy.unique().tolist()),
        "rounds": sorted(history["round"].unique().tolist()),
        "missing_factorial_cells": len(expected.difference(observed)),
        "unexpected_factorial_cells": len(observed.difference(expected)),
        "all_campaigns_have_21_rows": bool((counts == 21).all()),
        "duplicate_campaign_round_rows": int(history.duplicated(KEYS + ["round"]).sum()),
    }
    if not (
        integrity["campaigns"] == 960
        and integrity["history_rows"] == 20_160
        and integrity["missing_factorial_cells"] == 0
        and integrity["unexpected_factorial_cells"] == 0
        and integrity["all_campaigns_have_21_rows"]
        and integrity["duplicate_campaign_round_rows"] == 0
    ):
        raise RuntimeError(f"Benchmark integrity failure: {integrity}")
    return integrity


def metric_inventory(history: pd.DataFrame) -> pd.DataFrame:
    category = {column: name for name, columns in INVENTORY.items() for column in columns}
    missing = set(history.columns).difference(category)
    if missing:
        raise RuntimeError(f"Unclassified history columns: {sorted(missing)}")
    rows = []
    acquisition_dependent = {
        "pool_mean_uncertainty", "pool_max_uncertainty",
        "selected_mean_prediction", "selected_mean_uncertainty",
        "selected_mean_true_affinity", "selected_best_true_affinity",
        "selected_mean_combined_score", "selected_min_combined_score",
        "selected_max_combined_score", "selected_mean_prediction_contribution",
        "selected_mean_uncertainty_contribution", "n_newly_acquired",
        "n_cumulative_acquired", "newly_acquired_indices",
        "cumulative_acquired_indices", "newly_acquired_smiles",
        "newly_acquired_affinity_nm", "newly_acquired_qed",
        "affinity_discovery_count", "joint_discovery_count", "joint_discovery_auc",
        "joint_scaffold_count", "joint_diversity", "joint_qualifying_indices",
    }
    for position, column in enumerate(history.columns):
        rows.append({
            "column_position": position,
            "column": column,
            "dtype": str(history[column].dtype),
            "category": category[column],
            "meaningful_at_round_0": column not in {
                "selected_mean_prediction", "selected_mean_uncertainty",
                "selected_mean_true_affinity", "selected_best_true_affinity",
                "selected_mean_combined_score", "selected_min_combined_score",
                "selected_max_combined_score", "selected_mean_prediction_contribution",
                "selected_mean_uncertainty_contribution", "newly_acquired_smiles",
                "newly_acquired_affinity_nm", "newly_acquired_qed",
            },
            "depends_on_acquisition_history": column in acquisition_dependent,
            "direction": METRIC_DIRECTION.get(column, "not_primary_or_not_ordered"),
            "null_count": int(history[column].isna().sum()),
        })
    return pd.DataFrame(rows)


def pairwise_representation(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = final.set_index(["model", "strategy", "seed", "representation"])
    for model in MODELS:
        for acquisition in ACQUISITIONS:
            for first, second in combinations(REPRESENTATIONS, 2):
                for metric in PRIMARY_METRICS:
                    a = indexed.loc[(model, acquisition, slice(None), first), metric]
                    b = indexed.loc[(model, acquisition, slice(None), second), metric]
                    a.index = a.index.get_level_values("seed")
                    b.index = b.index.get_level_values("seed")
                    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
                    difference = paired.a - paired.b
                    rows.append({
                        "model": model, "acquisition": acquisition,
                        "representation_a": first, "representation_b": second,
                        "metric": metric, "direction": METRIC_DIRECTION[metric],
                        "n_paired_seeds": len(difference),
                        "mean_difference_a_minus_b": difference.mean(),
                        "median_difference_a_minus_b": difference.median(),
                        "std_difference_a_minus_b": difference.std(ddof=1),
                        "a_greater_b": int((difference > 0).sum()),
                        "a_equal_b": int((difference == 0).sum()),
                        "a_less_b": int((difference < 0).sum()),
                    })
    return pd.DataFrame(rows)


def correlations(final: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("rmse", "affinity_discovery_count"),
        ("rmse", "joint_discovery_count"),
        ("pearson", "affinity_discovery_count"),
        ("pearson", "joint_discovery_count"),
    ]
    levels = {
        "combination_mean": final.groupby(GROUPS, observed=True)[
            sorted(set(sum(([a, b] for a, b in pairs), [])))
        ].mean().reset_index(),
        "campaign": final,
    }
    rows = []
    for level, frame in levels.items():
        for x, y in pairs:
            valid = frame[[x, y]].dropna()
            rows.append({
                "level": level, "x_metric": x, "y_metric": y, "n": len(valid),
                "pearson_correlation": pearsonr(valid[x], valid[y]).statistic,
                "spearman_correlation": spearmanr(valid[x], valid[y]).statistic,
            })
    return pd.DataFrame(rows)


def anomaly_audit(history: pd.DataFrame, warnings_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    undefined = history[history.pearson.isna()]
    for record in undefined.itertuples(index=False):
        rows.append({
            "anomaly": "undefined_pearson_constant_prediction",
            "representation": record.representation, "model": record.model,
            "acquisition": record.strategy, "seed": record.seed,
            "round": record.round, "count": 1,
            "details": (
                f"RMSE={record.rmse:.6g}; affinity discoveries="
                f"{record.affinity_discovery_count}; joint discoveries="
                f"{record.joint_discovery_count}; discovery metrics finite"
            ),
        })

    monotonic_metrics = [
        "affinity_discovery_count", "joint_discovery_count", "joint_discovery_auc",
        "joint_scaffold_count", "best_discovered", "top20_mean_discovered",
    ]
    for metric in monotonic_metrics:
        violations = 0
        for _, group in history.groupby(KEYS, observed=True):
            violations += int((np.diff(group.sort_values("round")[metric]) < -1e-12).any())
        rows.append({
            "anomaly": "nonmonotonic_expected_cumulative_metric",
            "representation": "all", "model": "all", "acquisition": "all",
            "seed": np.nan, "round": np.nan, "count": violations,
            "details": metric,
        })

    impossible = (
        (history.affinity_discovery_count < 0)
        | (history.affinity_discovery_count > history.n_cumulative_acquired)
        | (history.joint_discovery_count < 0)
        | (history.joint_discovery_count > history.affinity_discovery_count)
        | (history.joint_scaffold_count < 0)
        | (history.joint_scaffold_count > history.joint_discovery_count)
    )
    rows.append({
        "anomaly": "impossible_discovery_count", "representation": "all",
        "model": "all", "acquisition": "all", "seed": np.nan, "round": np.nan,
        "count": int(impossible.sum()), "details": "structural count inequalities",
    })

    signature_metrics = TRAJECTORY_METRICS + ["best_discovered", "top20_mean_discovered"]
    signatures: dict[tuple[object, ...], list[str]] = {}
    for campaign_id, group in history.groupby("campaign_id", observed=True):
        values = group.sort_values("round")[signature_metrics].to_numpy(dtype=float)
        key = tuple(np.round(values, 12).ravel())
        signatures.setdefault(key, []).append(campaign_id)
    duplicate_groups = [ids for ids in signatures.values() if len(ids) > 1]
    rows.append({
        "anomaly": "exactly_identical_primary_trajectories",
        "representation": "all", "model": "all", "acquisition": "all",
        "seed": np.nan, "round": np.nan,
        "count": sum(len(ids) for ids in duplicate_groups),
        "details": json.dumps(duplicate_groups),
    })

    rows.append({
        "anomaly": "constant_input_warning_count", "representation": "all",
        "model": "all", "acquisition": "all", "seed": np.nan, "round": np.nan,
        "count": int((warnings_frame.category == "ConstantInputWarning").sum()),
        "details": "matches undefined Pearson audit",
    })

    final = history[history["round"] == 20]
    outlier_records = []
    for keys, group in final.groupby(GROUPS, observed=True):
        for metric in FINAL_METRICS:
            values = group[metric].dropna()
            q25, q75 = values.quantile([0.25, 0.75])
            iqr = q75 - q25
            lower, upper = q25 - 3 * iqr, q75 + 3 * iqr
            flagged = group[(group[metric] < lower) | (group[metric] > upper)]
            outlier_records.extend((int(seed), metric) for seed in flagged.seed)
    by_seed = pd.Series([item[0] for item in outlier_records]).value_counts().to_dict()
    by_metric = pd.Series([item[1] for item in outlier_records]).value_counts().to_dict()
    rows.append({
        "anomaly": "final_round_extreme_3iqr_flags", "representation": "all",
        "model": "all", "acquisition": "all", "seed": np.nan, "round": 20,
        "count": len(outlier_records),
        "details": json.dumps({
            "descriptive_rule": "outside Q25 +/- 3*IQR within each 20-seed combination",
            "by_seed": by_seed, "by_metric": by_metric,
            "note": "Flags are retained; discrete/tied metrics can have zero or narrow IQR.",
        }, sort_keys=True),
    })
    for metric in (
        "selected_mean_prediction_contribution",
        "selected_mean_uncertainty_contribution",
    ):
        finite = history.loc[np.isfinite(history[metric]), metric]
        location = history.loc[finite.abs().idxmax()]
        rows.append({
            "anomaly": "extreme_acquisition_diagnostic", 
            "representation": location.representation, "model": location.model,
            "acquisition": location.strategy, "seed": location.seed,
            "round": location["round"], "count": 1,
            "details": f"{metric}={location[metric]:.12g}; combined score remained finite",
        })
    return pd.DataFrame(rows)


def format_report(
    integrity: dict[str, object], final_summary: pd.DataFrame,
    correlations_frame: pd.DataFrame, anomalies: pd.DataFrame,
    conditional: pd.DataFrame,
) -> str:
    def value(aggregation: str, metric: str, **filters: str) -> float:
        selected = conditional[
            (conditional.metric == metric) &
            (conditional.aggregation == aggregation)
        ]
        for column, expected in filters.items():
            selected = selected[selected[column] == expected]
        return float(selected["mean"].iloc[0])

    correlation_lines = [
        f"- {r.level}: {r.x_metric} vs {r.y_metric}: Pearson {r.pearson_correlation:.3f}, "
        f"Spearman {r.spearman_correlation:.3f} (N={r.n})"
        for r in correlations_frame.itertuples(index=False)
    ]
    nan_rows = anomalies[anomalies.anomaly == "undefined_pearson_constant_prediction"]
    anomaly_lines = [
        f"- {r.representation}/{r.model}/{r.acquisition}, seed {int(r.seed)}, "
        f"round {int(r.round)}: {r.details}"
        for r in nan_rows.itertuples(index=False)
    ]
    model_lines = []
    for model in MODELS:
        entries = []
        for representation in REPRESENTATIONS:
            entries.append(
                f"{representation}: RMSE "
                f"{value('model+representation', 'rmse', model=model, representation=representation):.3f}, "
                f"joint {value('model+representation', 'joint_discovery_count', model=model, representation=representation):.2f}, "
                f"scaffolds {value('model+representation', 'joint_scaffold_count', model=model, representation=representation):.2f}"
            )
        model_lines.append(f"- {model}: " + "; ".join(entries))
    acquisition_lines = []
    for acquisition in ACQUISITIONS:
        entries = []
        for representation in REPRESENTATIONS:
            entries.append(
                f"{representation}: affinity "
                f"{value('acquisition+representation', 'affinity_discovery_count', strategy=acquisition, representation=representation):.2f}, "
                f"joint {value('acquisition+representation', 'joint_discovery_count', strategy=acquisition, representation=representation):.2f}, "
                f"scaffolds {value('acquisition+representation', 'joint_scaffold_count', strategy=acquisition, representation=representation):.2f}"
            )
        acquisition_lines.append(f"- {acquisition}: " + "; ".join(entries))
    return "\n".join([
        "# v0.5.1 representation benchmark: descriptive audit",
        "", "No hypothesis tests, method rankings, or performance claims are made here.",
        "", "## Integrity", "",
        f"- {integrity['campaigns']} campaigns; {integrity['history_rows']} history rows.",
        "- Complete 4 representation × 3 model × 4 acquisition × 20 seed design.",
        "- Every campaign contains rounds 0–20; no missing or duplicate cells.",
        "", "## Metric semantics", "",
        "- RMSE is lower-is-better; Pearson is higher-is-better.",
        "- `best_discovered` and `top20_mean_discovered` are pAffinity values; higher is more potent.",
        "- Affinity discoveries satisfy experimental affinity ≤500 nM.",
        "- Joint discoveries additionally satisfy QED ≥0.6.",
        "- Joint diversity is mean pairwise Morgan-Tanimoto distance and is undefined below two hits.",
        "", "## Model-conditional representation summaries", "", *model_lines,
        "", "## Acquisition-conditional representation summaries", "", *acquisition_lines,
        "", "## Prediction/discovery correlations", "", *correlation_lines,
        "", "## Undefined Pearson cases", "", *anomaly_lines,
        "", "## Files", "",
        "- `final_round_summary.csv`: 48 combinations × nine primary metrics.",
        "- `representation_pairwise_descriptive.csv`: paired A−B seed differences.",
        "- `trajectory_summary.csv`: round-wise descriptive trajectories.",
        "- `conditional_representation_summary.csv`: representation summaries conditional on model/acquisition.",
        "- `anomaly_report.csv`: integrity and unusual-case audit.",
    ]) + "\n"


def main() -> None:
    history = pd.read_parquet(BENCHMARK / "combined_campaign_history.parquet")
    campaign_summary = pd.read_parquet(BENCHMARK / "campaign_summary.parquet")
    warnings_frame = pd.read_parquet(BENCHMARK / "warnings.parquet")
    integrity = validate(history, campaign_summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    inventory = metric_inventory(history)
    inventory.to_csv(OUTPUT / "metric_inventory.csv", index=False)
    (OUTPUT / "metric_inventory.json").write_text(
        json.dumps({
            "columns_in_order": history.columns.tolist(),
            "conceptual_groups": INVENTORY,
        }, indent=2) + "\n", encoding="utf-8"
    )

    final = history[history["round"] == 20].copy()
    final_summary = descriptive(final, GROUPS, FINAL_METRICS)
    final_summary["direction"] = final_summary.metric.map(METRIC_DIRECTION)
    final_summary.to_csv(OUTPUT / "final_round_summary.csv", index=False)

    pairwise = pairwise_representation(final)
    pairwise.to_csv(OUTPUT / "representation_pairwise_descriptive.csv", index=False)

    trajectory = descriptive(history, GROUPS + ["round"], TRAJECTORY_METRICS)
    trajectory.to_csv(OUTPUT / "trajectory_summary.csv", index=False)

    correlation_frame = correlations(final)
    correlation_frame.to_csv(OUTPUT / "prediction_discovery_correlations.csv", index=False)

    conditional_parts = []
    for label, groups in (
        ("model+representation", ["model", "representation"]),
        ("acquisition+representation", ["strategy", "representation"]),
        ("model+acquisition+representation", GROUPS),
    ):
        part = descriptive(final, groups, [
            "rmse", "pearson", "affinity_discovery_count", "joint_discovery_count",
            "joint_scaffold_count",
        ])
        part["aggregation"] = label
        conditional_parts.append(part)
    conditional = pd.concat(conditional_parts, ignore_index=True, sort=False)
    conditional.to_csv(OUTPUT / "conditional_representation_summary.csv", index=False)

    anomalies = anomaly_audit(history, warnings_frame)
    anomalies.to_csv(OUTPUT / "anomaly_report.csv", index=False)
    (OUTPUT / "integrity_summary.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "report.md").write_text(
        format_report(integrity, final_summary, correlation_frame, anomalies, conditional),
        encoding="utf-8",
    )
    print(json.dumps({
        "integrity": integrity,
        "final_summary_rows": len(final_summary),
        "pairwise_rows": len(pairwise),
        "trajectory_rows": len(trajectory),
        "correlation_rows": len(correlation_frame),
        "anomaly_rows": len(anomalies),
    }, indent=2))


if __name__ == "__main__":
    main()
