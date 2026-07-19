from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.targets import load_target_regression
from src.regression.simulation import (
    run_regression_simulation,
)


DEFAULT_BUDGETS = [
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
]

DEFAULT_STRATEGIES = [
    "random",
    "ucb",
]

LEARNING_CURVE_METRICS = [
    "rmse",
    "mae",
    "r2",
    "pearson",
    "best_discovered",
    "top20_mean_discovered",
    "mean_discovered",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure active-learning performance as a function "
            "of the number of labelled molecules."
        )
    )

    parser.add_argument(
        "--target",
        default="EGFR",
        help="Regression target to load.",
    )

    parser.add_argument(
        "--model",
        default="random_forest",
        help="Regression surrogate to benchmark.",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "random",
            "greedy",
            "uncertainty",
            "ucb",
            "uncertainty_diverse",
        ],
        default=DEFAULT_STRATEGIES,
        help="Acquisition strategies to compare.",
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
        help=(
            "Total labelled-set sizes at which performance "
            "will be extracted."
        ),
    )

    parser.add_argument(
        "--n-initial",
        type=int,
        default=20,
        help="Initial labelled-set size.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of molecules acquired per round.",
    )

    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=100,
        help=(
            "Candidate pool size used by strategies that "
            "require one."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved as a fixed evaluation set.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Exploration weight used by UCB.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of paired campaign seeds.",
    )

    parser.add_argument(
        "--supervised-summary",
        type=Path,
        default=None,
        help=(
            "Optional summary.csv from supervised_benchmark.py. "
            "Used to draw the supervised-performance ceiling."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for benchmark outputs. A timestamped "
            "directory is created by default."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
    n_samples: int,
) -> list[int]:
    if args.seeds < 1:
        raise ValueError(
            "--seeds must be at least 1."
        )

    if args.n_initial < 2:
        raise ValueError(
            "--n-initial must be at least 2."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1."
        )

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(
            "--test-size must be between 0 and 1."
        )

    if args.beta < 0:
        raise ValueError(
            "--beta must be non-negative."
        )

    budgets = sorted(set(args.budgets))

    if not budgets:
        raise ValueError(
            "At least one label budget is required."
        )

    for budget in budgets:
        if budget < args.n_initial:
            raise ValueError(
                f"Budget {budget} is smaller than the initial "
                f"labelled-set size {args.n_initial}."
            )

        difference = budget - args.n_initial

        if difference % args.batch_size != 0:
            raise ValueError(
                f"Budget {budget} is not reachable from "
                f"n_initial={args.n_initial} with "
                f"batch_size={args.batch_size}."
            )

    approximate_pool_size = int(
        np.floor(
            n_samples * (1.0 - args.test_size)
        )
    )

    if budgets[-1] > approximate_pool_size:
        raise ValueError(
            f"Maximum budget {budgets[-1]} exceeds the "
            f"approximate training pool size "
            f"{approximate_pool_size}."
        )

    if (
        "ucb" in args.strategies
        and args.beta is None
    ):
        raise ValueError(
            "--beta is required when UCB is selected."
        )

    return budgets


def budget_to_round(
    budget: int,
    n_initial: int,
    batch_size: int,
) -> int:
    return (
        budget - n_initial
    ) // batch_size


def build_budget_round_map(
    budgets: list[int],
    n_initial: int,
    batch_size: int,
) -> dict[int, int]:
    return {
        budget: budget_to_round(
            budget=budget,
            n_initial=n_initial,
            batch_size=batch_size,
        )
        for budget in budgets
    }


def extract_budget_rows(
    campaign_history: pd.DataFrame,
    budgets: list[int],
) -> pd.DataFrame:
    available_sizes = set(
        campaign_history["n_labeled"]
        .astype(int)
        .tolist()
    )

    missing_budgets = [
        budget
        for budget in budgets
        if budget not in available_sizes
    ]

    if missing_budgets:
        raise RuntimeError(
            "Campaign did not reach the requested budgets: "
            f"{missing_budgets}. Available labelled sizes: "
            f"{sorted(available_sizes)}"
        )

    selected = campaign_history[
        campaign_history["n_labeled"].isin(
            budgets
        )
    ].copy()

    selected["label_budget"] = (
        selected["n_labeled"].astype(int)
    )

    return selected


def summarize_learning_curve(
    history: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "target",
        "model",
        "strategy",
        "seed",
        "label_budget",
        *LEARNING_CURVE_METRICS,
    }

    missing = sorted(
        required_columns.difference(
            history.columns
        )
    )

    if missing:
        raise ValueError(
            "Learning-curve history is missing columns: "
            + ", ".join(missing)
        )

    duplicate_mask = history.duplicated(
        subset=[
            "model",
            "strategy",
            "seed",
            "label_budget",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = history.loc[
            duplicate_mask,
            [
                "model",
                "strategy",
                "seed",
                "label_budget",
            ],
        ]

        raise ValueError(
            "Duplicate campaign checkpoints found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    named_aggregations: dict[
        str,
        pd.NamedAgg,
    ] = {
        "n_seeds": pd.NamedAgg(
            column="seed",
            aggfunc="nunique",
        )
    }

    for metric in LEARNING_CURVE_METRICS:
        named_aggregations[
            f"{metric}_mean"
        ] = pd.NamedAgg(
            column=metric,
            aggfunc="mean",
        )

        named_aggregations[
            f"{metric}_std"
        ] = pd.NamedAgg(
            column=metric,
            aggfunc="std",
        )

        named_aggregations[
            f"{metric}_median"
        ] = pd.NamedAgg(
            column=metric,
            aggfunc="median",
        )

    summary = (
        history.groupby(
            [
                "target",
                "model",
                "strategy",
                "label_budget",
            ],
            sort=True,
        )
        .agg(**named_aggregations)
        .reset_index()
        .sort_values(
            [
                "model",
                "strategy",
                "label_budget",
            ]
        )
        .reset_index(drop=True)
    )

    return summary


def load_supervised_ceiling(
    summary_path: Path | None,
    model_name: str,
) -> dict[str, float]:
    if summary_path is None:
        return {}

    if not summary_path.exists():
        raise FileNotFoundError(
            "Supervised summary not found: "
            f"{summary_path}"
        )

    supervised = pd.read_csv(
        summary_path
    )

    required_columns = {
        "model",
        "rmse_mean",
        "mae_mean",
        "r2_mean",
        "pearson_mean",
    }

    missing = sorted(
        required_columns.difference(
            supervised.columns
        )
    )

    if missing:
        raise ValueError(
            "Supervised summary is missing columns: "
            + ", ".join(missing)
        )

    model_rows = supervised[
        supervised["model"] == model_name
    ]

    if len(model_rows) != 1:
        available_models = sorted(
            supervised["model"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Expected one supervised row for model "
            f"'{model_name}', found {len(model_rows)}. "
            f"Available models: {available_models}"
        )

    row = model_rows.iloc[0]

    return {
        "rmse": float(row["rmse_mean"]),
        "mae": float(row["mae_mean"]),
        "r2": float(row["r2_mean"]),
        "pearson": float(
            row["pearson_mean"]
        ),
    }


def make_json_serializable(
    value: Any,
) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(
                item
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return value


def save_config(
    config: dict[str, Any],
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(config),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


def plot_metric_learning_curve(
    summary: pd.DataFrame,
    metric: str,
    supervised_ceiling: float | None,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    for strategy, group in summary.groupby(
        "strategy",
        sort=True,
    ):
        group = group.sort_values(
            "label_budget"
        )

        x = group[
            "label_budget"
        ].to_numpy()

        mean = group[
            f"{metric}_mean"
        ].to_numpy()

        std = group[
            f"{metric}_std"
        ].fillna(0.0).to_numpy()

        axis.plot(
            x,
            mean,
            marker="o",
            label=strategy,
        )

        axis.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    if supervised_ceiling is not None:
        axis.axhline(
            supervised_ceiling,
            linestyle="--",
            linewidth=1.5,
            label=(
                "Supervised CV ceiling "
                f"({supervised_ceiling:.3f})"
            ),
        )

    axis.set_xlabel(
        "Number of labelled molecules"
    )

    axis.set_ylabel(
        metric.replace(
            "_",
            " ",
        ).title()
    )

    axis.set_title(
        f"{metric.replace('_', ' ').title()} "
        "learning curve"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_outputs(
    history: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
    supervised_ceiling: dict[str, float],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_columns = [
        "target",
        "model",
        "strategy",
        "beta",
        "seed",
        "round",
        "label_budget",
        "n_labeled",
        *LEARNING_CURVE_METRICS,
    ]

    history_columns = [
        column
        for column in history_columns
        if column in history.columns
    ]

    remaining_columns = [
        column
        for column in history.columns
        if column not in history_columns
    ]

    ordered_history = history[
        history_columns
        + remaining_columns
    ].copy()

    paths = {
        "history": (
            output_dir
            / "learning_curve_history.csv"
        ),
        "summary": (
            output_dir
            / "learning_curve_summary.csv"
        ),
        "config": (
            output_dir
            / "config.json"
        ),
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
        "n_history_rows": int(
            len(ordered_history)
        ),
        "n_summary_rows": int(
            len(summary)
        ),
        "n_strategies": int(
            ordered_history[
                "strategy"
            ].nunique()
        ),
        "n_seeds": int(
            ordered_history[
                "seed"
            ].nunique()
        ),
        "supervised_ceiling": (
            supervised_ceiling
        ),
        "metrics": (
            LEARNING_CURVE_METRICS
        ),
    }

    save_config(
        config=config_with_metadata,
        output_path=paths["config"],
    )

    for metric in [
        "pearson",
        "rmse",
        "r2",
        "best_discovered",
        "top20_mean_discovered",
    ]:
        figure_path = (
            output_dir
            / f"{metric}_learning_curve.png"
        )

        plot_metric_learning_curve(
            summary=summary,
            metric=metric,
            supervised_ceiling=(
                supervised_ceiling.get(
                    metric
                )
            ),
            output_path=figure_path,
        )

        paths[
            f"{metric}_figure"
        ] = figure_path

    return paths


def main() -> None:
    args = parse_args()

    X, y = load_target_regression(
        args.target
    )

    X = np.asarray(X)
    y = np.asarray(y)

    budgets = validate_arguments(
        args=args,
        n_samples=len(y),
    )

    budget_round_map = (
        build_budget_round_map(
            budgets=budgets,
            n_initial=args.n_initial,
            batch_size=args.batch_size,
        )
    )

    max_budget = max(
        budgets
    )

    max_rounds = budget_to_round(
        budget=max_budget,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
    )

    supervised_ceiling = (
        load_supervised_ceiling(
            summary_path=(
                args.supervised_summary
            ),
            model_name=args.model,
        )
    )

    campaign_checkpoints: list[
        pd.DataFrame
    ] = []

    for strategy in args.strategies:
        print(
            f"\nStrategy: {strategy}"
        )

        for seed in range(
            args.seeds
        ):
            print(
                f"  Seed {seed + 1}/"
                f"{args.seeds}",
                end="\r",
            )

            campaign = (
                run_regression_simulation(
                    X=X,
                    y=y,
                    model_name=args.model,
                    strategy=strategy,
                    random_state=seed,
                    n_initial=args.n_initial,
                    batch_size=args.batch_size,
                    n_rounds=max_rounds,
                    test_size=args.test_size,
                    candidate_pool_size=(
                        args.candidate_pool_size
                    ),
                    beta=(
                        args.beta
                        if strategy == "ucb"
                        else None
                    ),
                )
                .copy()
            )

            campaign[
                "target"
            ] = args.target

            campaign[
                "model"
            ] = args.model

            checkpoints = (
                extract_budget_rows(
                    campaign_history=campaign,
                    budgets=budgets,
                )
            )

            campaign_checkpoints.append(
                checkpoints
            )

        print(
            f"  Completed {args.seeds} "
            "seeds."
        )

    history = pd.concat(
        campaign_checkpoints,
        ignore_index=True,
    )

    summary = summarize_learning_curve(
        history
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = Path(
            "results/regression/"
            "learning_curves"
        ) / (
            f"{args.target.lower()}_"
            f"{args.model}_"
            f"{timestamp}"
        )
    else:
        output_dir = args.output_dir

    config = {
        "target": args.target,
        "model": args.model,
        "strategies": args.strategies,
        "budgets": budgets,
        "budget_round_map": (
            budget_round_map
        ),
        "n_initial": args.n_initial,
        "batch_size": args.batch_size,
        "maximum_rounds": max_rounds,
        "candidate_pool_size": (
            args.candidate_pool_size
        ),
        "test_size": args.test_size,
        "beta": args.beta,
        "seeds": args.seeds,
        "supervised_summary": (
            args.supervised_summary
        ),
        "n_samples": int(
            len(y)
        ),
        "n_features": int(
            X.shape[1]
        ),
    }

    paths = save_outputs(
        history=history,
        summary=summary,
        config=config,
        supervised_ceiling=(
            supervised_ceiling
        ),
        output_dir=output_dir,
    )

    print(
        f"\nSaved learning-curve benchmark to:"
        f"\n{output_dir}"
    )

    print(
        "\nGenerated files:"
    )

    for name, path in paths.items():
        print(
            f"  {name:28s} {path}"
        )

    print(
        "\nRows per strategy and budget:"
    )

    print(
        history.groupby(
            [
                "strategy",
                "label_budget",
            ]
        )
        .size()
        .to_string()
    )

    display_columns = [
        "strategy",
        "label_budget",
        "n_seeds",
        "pearson_mean",
        "pearson_std",
        "rmse_mean",
        "rmse_std",
        "best_discovered_mean",
        "top20_mean_discovered_mean",
    ]

    print(
        "\nLearning-curve summary:"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    main()