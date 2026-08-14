from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SUMMARY_METRICS = [
    "rmse",
    "mae",
    "r2",
    "pearson",
    "best_discovered",
    "top20_mean_discovered",
    "mean_discovered",
]

IDENTIFIER_COLUMNS = [
    "target",
    "model",
    "strategy",
    "beta",
    "seed",
    "round",
    "n_labeled",
]


def validate_benchmark_history(
    history: pd.DataFrame,
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> None:
    """
    Validate that a benchmark history contains the columns required
    for summarization and paired model comparisons.

    Parameters
    ----------
    history
        Complete benchmark history with one row per
        model, seed, and active-learning round.
    metrics
        Metrics expected in the benchmark history.

    Raises
    ------
    ValueError
        If the history is empty, required columns are missing,
        duplicate model/seed/round rows are found, or different
        models contain incompatible seed-round combinations.
    """
    if history.empty:
        raise ValueError("Benchmark history is empty.")

    required_columns = {
        "model",
        "seed",
        "round",
        *metrics,
    }

    missing_columns = sorted(
        required_columns.difference(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Benchmark history is missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_mask = history.duplicated(
        subset=["model", "seed", "round"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = history.loc[
            duplicate_mask,
            ["model", "seed", "round"],
        ]

        raise ValueError(
            "Duplicate model/seed/round rows found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    model_seed_rounds = {
        model: set(
            zip(
                group["seed"].tolist(),
                group["round"].tolist(),
            )
        )
        for model, group in history.groupby("model")
    }

    reference_model = next(iter(model_seed_rounds))
    reference_pairs = model_seed_rounds[reference_model]

    for model, seed_round_pairs in model_seed_rounds.items():
        if seed_round_pairs != reference_pairs:
            raise ValueError(
                "Models do not contain matching seed-round combinations. "
                f"Reference model '{reference_model}' has "
                f"{len(reference_pairs)} combinations, while '{model}' "
                f"has {len(seed_round_pairs)}."
            )


def get_final_round_rows(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the final available round independently for each
    model and seed.

    This is safer than using the global maximum round because it
    remains correct if an incomplete campaign is ever present.
    """
    required_columns = {
        "model",
        "seed",
        "round",
    }

    missing_columns = sorted(
        required_columns.difference(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Cannot identify final rounds. Missing columns: "
            + ", ".join(missing_columns)
        )

    final_indices = (
        history.groupby(
            ["model", "seed"],
            sort=False,
        )["round"]
        .idxmax()
    )

    final_rows = (
        history.loc[final_indices]
        .sort_values(["model", "seed"])
        .reset_index(drop=True)
    )

    return final_rows


def summarize_final_round(
    history: pd.DataFrame,
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> pd.DataFrame:
    """
    Summarize final-round model performance across seeds.

    Returns one row per model with mean, sample standard deviation,
    median, minimum, and maximum for each requested metric.
    """
    validate_benchmark_history(
        history=history,
        metrics=metrics,
    )

    final_rows = get_final_round_rows(history)

    aggregations: dict[str, Any] = {
        "seed": pd.NamedAgg(
            column="seed",
            aggfunc="nunique",
        )
    }

    for metric in metrics:
        aggregations[f"{metric}_mean"] = pd.NamedAgg(
            column=metric,
            aggfunc="mean",
        )
        aggregations[f"{metric}_std"] = pd.NamedAgg(
            column=metric,
            aggfunc="std",
        )
        aggregations[f"{metric}_median"] = pd.NamedAgg(
            column=metric,
            aggfunc="median",
        )
        aggregations[f"{metric}_min"] = pd.NamedAgg(
            column=metric,
            aggfunc="min",
        )
        aggregations[f"{metric}_max"] = pd.NamedAgg(
            column=metric,
            aggfunc="max",
        )

    summary = (
        final_rows.groupby(
            "model",
            sort=True,
        )
        .agg(**aggregations)
        .rename(columns={"seed": "n_seeds"})
        .reset_index()
    )

    return summary


def build_paired_comparison(
    history: pd.DataFrame,
    reference_model: str,
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> pd.DataFrame:
    """
    Build paired final-round comparisons against a reference model.

    One row is produced for every:

        comparison model × seed × metric

    The difference convention is always:

        comparison_value - reference_value

    Therefore, interpretation depends on the metric. For example:

    - negative RMSE difference favours the comparison model;
    - positive R² or Pearson difference favours the comparison model.

    Parameters
    ----------
    history
        Complete benchmark history.
    reference_model
        Model used as the comparison baseline.
    metrics
        Final-round metrics to compare.

    Returns
    -------
    pd.DataFrame
        Long-format paired comparison table.
    """
    validate_benchmark_history(
        history=history,
        metrics=metrics,
    )

    final_rows = get_final_round_rows(history)

    available_models = sorted(
        final_rows["model"].unique()
    )

    if reference_model not in available_models:
        raise ValueError(
            f"Reference model '{reference_model}' is not present. "
            f"Available models: {available_models}"
        )

    comparison_models = [
        model
        for model in available_models
        if model != reference_model
    ]

    if not comparison_models:
        return pd.DataFrame(
            columns=[
                "seed",
                "metric",
                "reference_model",
                "comparison_model",
                "reference_value",
                "comparison_value",
                "difference",
            ]
        )

    reference = (
        final_rows[
            final_rows["model"] == reference_model
        ]
        .set_index("seed")
        .sort_index()
    )

    rows: list[dict[str, Any]] = []

    for comparison_model in comparison_models:
        comparison = (
            final_rows[
                final_rows["model"] == comparison_model
            ]
            .set_index("seed")
            .sort_index()
        )

        common_seeds = reference.index.intersection(
            comparison.index
        )

        missing_reference_seeds = comparison.index.difference(
            reference.index
        )
        missing_comparison_seeds = reference.index.difference(
            comparison.index
        )

        if (
            len(missing_reference_seeds) > 0
            or len(missing_comparison_seeds) > 0
        ):
            raise ValueError(
                "Paired comparison requires matching seeds. "
                f"Reference-only seeds: "
                f"{missing_comparison_seeds.tolist()}; "
                f"comparison-only seeds: "
                f"{missing_reference_seeds.tolist()}."
            )

        for seed in common_seeds:
            for metric in metrics:
                reference_value = reference.at[seed, metric]
                comparison_value = comparison.at[seed, metric]

                if pd.isna(reference_value) or pd.isna(
                    comparison_value
                ):
                    difference = np.nan
                else:
                    difference = (
                        float(comparison_value)
                        - float(reference_value)
                    )

                rows.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "reference_model": reference_model,
                        "comparison_model": comparison_model,
                        "reference_value": reference_value,
                        "comparison_value": comparison_value,
                        "difference": difference,
                    }
                )

    paired = pd.DataFrame(rows)

    return paired.sort_values(
        [
            "comparison_model",
            "metric",
            "seed",
        ]
    ).reset_index(drop=True)


def summarize_paired_comparison(
    paired_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize paired differences across seeds.

    Returns one row per comparison-model and metric combination.
    """
    required_columns = {
        "metric",
        "reference_model",
        "comparison_model",
        "difference",
    }

    missing_columns = sorted(
        required_columns.difference(
            paired_comparison.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Paired comparison is missing required columns: "
            + ", ".join(missing_columns)
        )

    if paired_comparison.empty:
        return pd.DataFrame(
            columns=[
                "reference_model",
                "comparison_model",
                "metric",
                "n_pairs",
                "difference_mean",
                "difference_std",
                "difference_median",
                "difference_min",
                "difference_max",
            ]
        )

    summary = (
        paired_comparison.groupby(
            [
                "reference_model",
                "comparison_model",
                "metric",
            ],
            sort=True,
        )
        .agg(
            n_pairs=("difference", "count"),
            difference_mean=("difference", "mean"),
            difference_std=("difference", "std"),
            difference_median=("difference", "median"),
            difference_min=("difference", "min"),
            difference_max=("difference", "max"),
        )
        .reset_index()
    )

    return summary


def order_history_columns(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Put stable identifier columns first while preserving all
    additional diagnostic columns.
    """
    leading_columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in history.columns
    ]

    remaining_columns = [
        column
        for column in history.columns
        if column not in leading_columns
    ]

    return history[
        leading_columns + remaining_columns
    ].copy()


def make_json_serializable(
    value: Any,
) -> Any:
    """
    Convert common Python, NumPy, and pathlib values into objects
    accepted by json.dump.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return value


def save_benchmark_config(
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save the benchmark configuration as formatted JSON.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_config = make_json_serializable(
        config
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_config,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")



ROUND_SUMMARY_METRICS = [
    *SUMMARY_METRICS,
    "fraction_best_found",
    "distance_to_best",
    "selected_mean_prediction",
    "selected_mean_uncertainty",
    "selected_mean_true_affinity",
    "selected_best_true_affinity",
    "selected_mean_combined_score",
    "selected_min_combined_score",
    "selected_max_combined_score",
    "selected_mean_prediction_contribution",
    "selected_mean_uncertainty_contribution",
    "pool_mean_uncertainty",
    "pool_max_uncertainty",
    "round_runtime_seconds",
]


def summarize_by_round(
    history: pd.DataFrame,
    metrics: Sequence[str] = ROUND_SUMMARY_METRICS,
) -> pd.DataFrame:
    """Summarize each active-learning round across random seeds."""
    required_columns = {"model", "seed", "round"}
    missing_columns = sorted(
        required_columns.difference(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Cannot summarize benchmark rounds. Missing columns: "
            + ", ".join(missing_columns)
        )

    available_metrics = [
        metric for metric in metrics
        if metric in history.columns
    ]

    if not available_metrics:
        raise ValueError(
            "No requested round-summary metrics are present in history."
        )

    group_columns = [
        column
        for column in [
            "target",
            "model",
            "strategy",
            "beta",
            "round",
        ]
        if column in history.columns
    ]

    aggregations: dict[str, Any] = {
        "n_seeds": pd.NamedAgg(
            column="seed",
            aggfunc="nunique",
        )
    }

    for metric in available_metrics:
        aggregations[f"{metric}_mean"] = pd.NamedAgg(
            column=metric,
            aggfunc="mean",
        )
        aggregations[f"{metric}_std"] = pd.NamedAgg(
            column=metric,
            aggfunc="std",
        )

    return (
        history.groupby(
            group_columns,
            sort=True,
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )

def save_benchmark_outputs(
    history: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
    reference_model: str | None = None,
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> dict[str, Path]:
    """
    Validate, summarize, and save all stable benchmark outputs.

    Files produced
    --------------
    history.csv
        Complete model × seed × round history.

    final_round_summary.csv
        Final-round descriptive statistics by model.

    round_summary.csv
        Round-by-round means and standard deviations across seeds.

    paired_comparison.csv
        Seed-level paired differences against the reference model.

    paired_comparison_summary.csv
        Aggregate paired differences across seeds.

    config.json
        Configuration used to generate the benchmark.

    Parameters
    ----------
    history
        Complete benchmark history.
    output_dir
        Directory in which benchmark files will be written.
    config
        Benchmark configuration.
    reference_model
        Baseline model for paired comparisons. If omitted, the first
        model in config["models"] is used when possible.
    metrics
        Metrics included in summaries and comparisons.

    Returns
    -------
    dict[str, Path]
        Paths of all generated files.
    """
    validate_benchmark_history(
        history=history,
        metrics=metrics,
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_history = order_history_columns(
        history
    )

    if reference_model is None:
        configured_models = config.get("models", [])

        if configured_models:
            reference_model = configured_models[0]
        else:
            reference_model = sorted(
                ordered_history["model"].unique()
            )[0]

    summary = summarize_final_round(
        history=ordered_history,
        metrics=metrics,
    )

    paired = build_paired_comparison(
        history=ordered_history,
        reference_model=reference_model,
        metrics=metrics,
    )

    paired_summary = summarize_paired_comparison(
        paired
    )

    round_summary = summarize_by_round(
        ordered_history
    )

    paths = {
        "history": output_dir / "history.csv",
        "final_round_summary": (
            output_dir / "final_round_summary.csv"
        ),
        "round_summary": (
            output_dir / "round_summary.csv"
        ),
        "paired_comparison": (
            output_dir / "paired_comparison.csv"
        ),
        "paired_comparison_summary": (
            output_dir
            / "paired_comparison_summary.csv"
        ),
        "config": output_dir / "config.json",
    }

    ordered_history.to_csv(
        paths["history"],
        index=False,
    )

    summary.to_csv(
        paths["final_round_summary"],
        index=False,
    )

    round_summary.to_csv(
        paths["round_summary"],
        index=False,
    )

    paired.to_csv(
        paths["paired_comparison"],
        index=False,
    )

    paired_summary.to_csv(
        paths["paired_comparison_summary"],
        index=False,
    )

    config_with_metadata = {
        **config,
        "reference_model": reference_model,
        "summary_metrics": list(metrics),
        "n_history_rows": len(ordered_history),
        "n_models": int(
            ordered_history["model"].nunique()
        ),
        "n_seeds": int(
            ordered_history["seed"].nunique()
        ),
        "max_round": int(
            ordered_history["round"].max()
        ),
    }

    save_benchmark_config(
        config=config_with_metadata,
        output_path=paths["config"],
    )

    return paths


def summarize_supervised_cv(
    history: pd.DataFrame,
    metrics: Sequence[str] = (
        "rmse",
        "mae",
        "r2",
        "pearson",
    ),
) -> pd.DataFrame:
    """
    Summarize repeated cross-validation results across folds.

    Returns one row per model with descriptive statistics for
    each requested metric.
    """
    if history.empty:
        raise ValueError(
            "Supervised benchmark history is empty."
        )

    required_columns = {
        "model",
        "repeat",
        "fold",
        *metrics,
    }

    missing_columns = sorted(
        required_columns.difference(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Supervised benchmark history is missing columns: "
            + ", ".join(missing_columns)
        )

    duplicate_mask = history.duplicated(
        subset=["model", "repeat", "fold"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = history.loc[
            duplicate_mask,
            ["model", "repeat", "fold"],
        ]

        raise ValueError(
            "Duplicate model/repeat/fold rows found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    aggregations: dict[str, Any] = {
        "n_evaluations": pd.NamedAgg(
            column="fold",
            aggfunc="count",
        )
    }

    for metric in metrics:
        aggregations[f"{metric}_mean"] = pd.NamedAgg(
            column=metric,
            aggfunc="mean",
        )
        aggregations[f"{metric}_std"] = pd.NamedAgg(
            column=metric,
            aggfunc="std",
        )
        aggregations[f"{metric}_median"] = pd.NamedAgg(
            column=metric,
            aggfunc="median",
        )
        aggregations[f"{metric}_min"] = pd.NamedAgg(
            column=metric,
            aggfunc="min",
        )
        aggregations[f"{metric}_max"] = pd.NamedAgg(
            column=metric,
            aggfunc="max",
        )

    return (
        history.groupby(
            "model",
            sort=True,
        )
        .agg(**aggregations)
        .reset_index()
    )


def save_supervised_benchmark_outputs(
    history: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
    metrics: Sequence[str] = (
        "rmse",
        "mae",
        "r2",
        "pearson",
    ),
) -> dict[str, Path]:
    """
    Save stable outputs from a supervised cross-validation benchmark.

    Files produced
    --------------
    history.csv
        One row per model, repeat, and fold.

    summary.csv
        Aggregate cross-validation performance by model.

    config.json
        Benchmark configuration and output metadata.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = summarize_supervised_cv(
        history=history,
        metrics=metrics,
    )

    ordered_columns = [
        "target",
        "model",
        "repeat",
        "fold",
        "split",
        "random_state",
        "n_train",
        "n_test",
        *metrics,
    ]

    ordered_columns = [
        column
        for column in ordered_columns
        if column in history.columns
    ]

    remaining_columns = [
        column
        for column in history.columns
        if column not in ordered_columns
    ]

    ordered_history = history[
        ordered_columns + remaining_columns
    ].copy()

    paths = {
        "history": output_dir / "history.csv",
        "summary": output_dir / "summary.csv",
        "config": output_dir / "config.json",
    }

    ordered_history.to_csv(
        paths["history"],
        index=False,
    )

    summary.to_csv(
        paths["summary"],
        index=False,
    )

    config_with_metadata = {
        **config,
        "n_history_rows": len(ordered_history),
        "n_models": int(
            ordered_history["model"].nunique()
        ),
        "n_evaluations_per_model": int(
            ordered_history.groupby("model")
            .size()
            .min()
        ),
        "summary_metrics": list(metrics),
    }

    save_benchmark_config(
        config=config_with_metadata,
        output_path=paths["config"],
    )

    return paths