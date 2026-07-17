import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.campaign_convergence import (
    evaluate_campaign_convergence,
)


DEFAULT_METRIC_CONFIG = {
    "rmse": {
        "direction": "minimize",
        "absolute_tolerance": 0.01,
        "relative_tolerance": 0.01,
    },
    "top20_mean_discovered": {
        "direction": "maximize",
        "absolute_tolerance": 0.05,
        "relative_tolerance": 0.01,
    },
    "best_discovered": {
        "direction": "maximize",
        "absolute_tolerance": 0.01,
        "relative_tolerance": 0.01,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-seed and aggregated convergence "
            "for saved active-learning campaign histories."
        )
    )

    parser.add_argument(
        "--history",
        required=True,
        help="Path to a campaign history CSV.",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRIC_CONFIG),
        help=(
            "Metrics to evaluate. Defaults to rmse, "
            "top20_mean_discovered, and best_discovered."
        ),
    )

    parser.add_argument(
        "--policy",
        choices=[
            "all",
            "any",
            "at_least_n",
        ],
        default="all",
    )

    parser.add_argument(
        "--minimum-converged",
        type=int,
        default=None,
        help=(
            "Required only when policy='at_least_n'."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--tolerance-logic",
        choices=[
            "all",
            "any",
        ],
        default="all",
    )

    parser.add_argument(
        "--maximum-iterations",
        type=int,
        default=None,
        help=(
            "Configured maximum campaign round. "
            "If omitted, the maximum round in the file is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="results/tables",
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional output prefix. By default, the history "
            "filename stem is used."
        ),
    )

    return parser.parse_args()


def validate_history(
    history: pd.DataFrame,
    metrics: list[str],
) -> None:
    required_columns = {
        "round",
        "seed",
        *metrics,
    }

    missing = required_columns - set(
        history.columns
    )

    if missing:
        raise ValueError(
            "History is missing required columns: "
            f"{sorted(missing)}"
        )

    if history.empty:
        raise ValueError(
            "History file is empty."
        )

    for metric in metrics:
        if not pd.api.types.is_numeric_dtype(
            history[metric]
        ):
            raise ValueError(
                f"Metric '{metric}' must be numeric."
            )

    if history[
        [
            "round",
            "seed",
            *metrics,
        ]
    ].isna().any().any():
        raise ValueError(
            "History contains missing values in required columns."
        )


def infer_group_columns(
    history: pd.DataFrame,
) -> list[str]:
    """
    Use available experiment identifiers without requiring every
    history file to contain exactly the same metadata columns.
    """
    preferred_columns = [
        "target",
        "strategy",
        "beta",
        "model",
        "representation",
        "seed",
    ]

    group_columns = [
        column
        for column in preferred_columns
        if column in history.columns
    ]

    if "seed" not in group_columns:
        raise ValueError(
            "History must contain a 'seed' column."
        )

    return group_columns


def build_metric_configuration(
    metrics: list[str],
) -> tuple[
    dict[str, str],
    dict[str, float | None],
    dict[str, float | None],
]:
    directions: dict[str, str] = {}
    absolute_tolerances: dict[
        str,
        float | None,
    ] = {}
    relative_tolerances: dict[
        str,
        float | None,
    ] = {}

    unknown_metrics = [
        metric
        for metric in metrics
        if metric not in DEFAULT_METRIC_CONFIG
    ]

    if unknown_metrics:
        raise ValueError(
            "No default convergence configuration exists for: "
            f"{unknown_metrics}. Add them to "
            "DEFAULT_METRIC_CONFIG before running."
        )

    for metric in metrics:
        config = DEFAULT_METRIC_CONFIG[
            metric
        ]

        directions[metric] = config[
            "direction"
        ]

        absolute_tolerances[metric] = (
            config[
                "absolute_tolerance"
            ]
        )

        relative_tolerances[metric] = (
            config[
                "relative_tolerance"
            ]
        )

    return (
        directions,
        absolute_tolerances,
        relative_tolerances,
    )


def metadata_from_group(
    group_columns: list[str],
    group_key,
) -> dict[str, object]:
    if len(group_columns) == 1:
        group_key = (
            group_key,
        )

    return dict(
        zip(
            group_columns,
            group_key,
        )
    )


def analyze_group(
    group: pd.DataFrame,
    *,
    metadata: dict[str, object],
    metrics: list[str],
    directions: dict[str, str],
    absolute_tolerances: dict[
        str,
        float | None,
    ],
    relative_tolerances: dict[
        str,
        float | None,
    ],
    patience: int,
    tolerance_logic: str,
    maximum_iterations: int,
    policy: str,
    minimum_converged: int | None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[pd.DataFrame],
]:
    group = (
        group.sort_values("round")
        .reset_index(drop=True)
    )

    if group["round"].duplicated().any():
        raise ValueError(
            "Each campaign group must contain one row per round. "
            f"Duplicate rounds found for {metadata}."
        )

    rounds = group[
        "round"
    ].to_numpy()

    metric_values = {
        metric: group[
            metric
        ].to_numpy()
        for metric in metrics
    }

    (
        campaign_result,
        metric_results,
    ) = evaluate_campaign_convergence(
        rounds=rounds,
        metric_values=metric_values,
        metric_directions=directions,
        absolute_tolerances=(
            absolute_tolerances
        ),
        relative_tolerances=(
            relative_tolerances
        ),
        patience=patience,
        tolerance_logic=(
            tolerance_logic
        ),
        maximum_iterations=(
            maximum_iterations
        ),
        policy=policy,
        minimum_converged=(
            minimum_converged
        ),
    )

    campaign_row = {
        **metadata,
        "n_metrics": (
            campaign_result.n_metrics
        ),
        "n_converged_metrics": (
            campaign_result
            .n_converged_metrics
        ),
        "convergence_fraction": (
            campaign_result
            .convergence_fraction
        ),
        "policy": campaign_result.policy,
        "minimum_converged": (
            campaign_result
            .minimum_converged
        ),
        "campaign_converged": (
            campaign_result
            .campaign_converged
        ),
        "converged_metrics": "|".join(
            campaign_result
            .converged_metrics
        ),
        "unconverged_metrics": "|".join(
            campaign_result
            .unconverged_metrics
        ),
        "maximum_iterations": (
            maximum_iterations
        ),
        "patience": patience,
        "tolerance_logic": (
            tolerance_logic
        ),
    }

    metric_rows = []

    round_tables = []

    for metric_name, result in (
        metric_results.items()
    ):
        metric_rows.append(
            {
                **metadata,
                "metric_name": metric_name,
                "direction": (
                    result.direction
                ),
                "currently_converged": (
                    result
                    .currently_converged
                ),
                "ever_converged": (
                    result.ever_converged
                ),
                "first_convergence_round": (
                    result
                    .first_convergence_round
                ),
                "current_plateau_start_round": (
                    result
                    .current_plateau_start_round
                ),
                "current_stable_rounds": (
                    result
                    .current_stable_rounds
                ),
                "best_round": (
                    result.best_round
                ),
                "best_value": (
                    result.best_value
                ),
                "final_round": (
                    result.final_round
                ),
                "reached_maximum_iterations": (
                    result
                    .reached_maximum_iterations
                ),
                "termination_status": (
                    result
                    .termination_status
                ),
                "absolute_tolerance": (
                    result
                    .absolute_tolerance
                ),
                "relative_tolerance": (
                    result
                    .relative_tolerance
                ),
                "patience": result.patience,
            }
        )

        round_table = (
            result.round_statistics.copy()
        )

        for key, value in metadata.items():
            round_table[key] = value

        round_table[
            "metric_name"
        ] = metric_name

        round_tables.append(
            round_table
        )

    return (
        campaign_row,
        metric_rows,
        round_tables,
    )


def aggregate_across_seeds(
    campaign_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_independent_columns = [
        column
        for column in [
            "target",
            "strategy",
            "beta",
            "model",
            "representation",
        ]
        if column
        in campaign_summary.columns
    ]

    if not seed_independent_columns:
        seed_independent_columns = [
            "__all_campaigns__"
        ]

        campaign_summary = (
            campaign_summary.assign(
                __all_campaigns__="all"
            )
        )

        metric_summary = (
            metric_summary.assign(
                __all_campaigns__="all"
            )
        )

    campaign_aggregate = (
        campaign_summary.groupby(
            seed_independent_columns,
            dropna=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            converged_seeds=(
                "campaign_converged",
                "sum",
            ),
            fraction_seeds_converged=(
                "campaign_converged",
                "mean",
            ),
            mean_convergence_fraction=(
                "convergence_fraction",
                "mean",
            ),
            std_convergence_fraction=(
                "convergence_fraction",
                "std",
            ),
        )
        .reset_index()
    )

    metric_group_columns = [
        *seed_independent_columns,
        "metric_name",
    ]

    metric_aggregate = (
        metric_summary.groupby(
            metric_group_columns,
            dropna=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            converged_seeds=(
                "currently_converged",
                "sum",
            ),
            fraction_seeds_converged=(
                "currently_converged",
                "mean",
            ),
            ever_converged_seeds=(
                "ever_converged",
                "sum",
            ),
            median_first_convergence_round=(
                "first_convergence_round",
                "median",
            ),
            minimum_first_convergence_round=(
                "first_convergence_round",
                "min",
            ),
            maximum_first_convergence_round=(
                "first_convergence_round",
                "max",
            ),
            median_best_round=(
                "best_round",
                "median",
            ),
            mean_best_value=(
                "best_value",
                "mean",
            ),
            std_best_value=(
                "best_value",
                "std",
            ),
        )
        .reset_index()
    )

    if (
        "__all_campaigns__"
        in campaign_aggregate.columns
    ):
        campaign_aggregate = (
            campaign_aggregate.drop(
                columns=[
                    "__all_campaigns__"
                ]
            )
        )

        metric_aggregate = (
            metric_aggregate.drop(
                columns=[
                    "__all_campaigns__"
                ]
            )
        )

    return (
        campaign_aggregate,
        metric_aggregate,
    )


def main() -> None:
    args = parse_args()

    history_path = Path(
        args.history
    )

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: "
            f"{history_path}"
        )

    history = pd.read_csv(
        history_path
    )

    validate_history(
        history,
        args.metrics,
    )

    group_columns = (
        infer_group_columns(
            history
        )
    )

    (
        directions,
        absolute_tolerances,
        relative_tolerances,
    ) = build_metric_configuration(
        args.metrics
    )

    maximum_iterations = (
        args.maximum_iterations
        if args.maximum_iterations
        is not None
        else int(
            history["round"].max()
        )
    )

    campaign_rows = []

    metric_rows = []

    round_tables = []

    grouped = history.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for group_key, group in grouped:
        metadata = metadata_from_group(
            group_columns,
            group_key,
        )

        (
            campaign_row,
            group_metric_rows,
            group_round_tables,
        ) = analyze_group(
            group,
            metadata=metadata,
            metrics=args.metrics,
            directions=directions,
            absolute_tolerances=(
                absolute_tolerances
            ),
            relative_tolerances=(
                relative_tolerances
            ),
            patience=args.patience,
            tolerance_logic=(
                args.tolerance_logic
            ),
            maximum_iterations=(
                maximum_iterations
            ),
            policy=args.policy,
            minimum_converged=(
                args.minimum_converged
            ),
        )

        campaign_rows.append(
            campaign_row
        )

        metric_rows.extend(
            group_metric_rows
        )

        round_tables.extend(
            group_round_tables
        )

    campaign_summary = pd.DataFrame(
        campaign_rows
    )

    metric_summary = pd.DataFrame(
        metric_rows
    )

    round_diagnostics = pd.concat(
        round_tables,
        ignore_index=True,
    )

    (
        campaign_aggregate,
        metric_aggregate,
    ) = aggregate_across_seeds(
        campaign_summary,
        metric_summary,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        args.output_prefix
        or history_path.stem
    )

    output_paths = {
        "campaign_summary": (
            output_dir
            / f"{prefix}_campaign_convergence.csv"
        ),
        "metric_summary": (
            output_dir
            / f"{prefix}_metric_convergence.csv"
        ),
        "round_diagnostics": (
            output_dir
            / f"{prefix}_round_convergence.csv"
        ),
        "campaign_aggregate": (
            output_dir
            / (
                f"{prefix}_campaign_convergence_"
                "aggregate.csv"
            )
        ),
        "metric_aggregate": (
            output_dir
            / (
                f"{prefix}_metric_convergence_"
                "aggregate.csv"
            )
        ),
    }

    campaign_summary.to_csv(
        output_paths[
            "campaign_summary"
        ],
        index=False,
    )

    metric_summary.to_csv(
        output_paths[
            "metric_summary"
        ],
        index=False,
    )

    round_diagnostics.to_csv(
        output_paths[
            "round_diagnostics"
        ],
        index=False,
    )

    campaign_aggregate.to_csv(
        output_paths[
            "campaign_aggregate"
        ],
        index=False,
    )

    metric_aggregate.to_csv(
        output_paths[
            "metric_aggregate"
        ],
        index=False,
    )

    print(
        "\nPer-seed campaign convergence"
    )

    display_columns = [
        column
        for column in [
            "target",
            "strategy",
            "beta",
            "seed",
            "n_converged_metrics",
            "n_metrics",
            "convergence_fraction",
            "campaign_converged",
        ]
        if column
        in campaign_summary.columns
    ]

    print(
        campaign_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nAcross-seed campaign summary"
    )

    print(
        campaign_aggregate.to_string(
            index=False
        )
    )

    print("\nSaved:")

    for path in output_paths.values():
        print(path)


if __name__ == "__main__":
    main()