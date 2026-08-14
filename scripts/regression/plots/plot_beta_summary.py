#!/usr/bin/env python3
"""
Plot a 2×2 comparison of UCB exploration parameters across regression models.

Each benchmark subdirectory is expected to contain:

    config.json
    final_round_summary.csv

By default, subdirectories are resolved relative to:

    results/regression/benchmarks

The script reads beta from each config.json and plots the mean ± standard
deviation reported in final_round_summary.csv.

Default panels
--------------
A) RMSE
B) Pearson correlation
C) Best affinity discovered
D) Mean affinity of the top 20 discovered molecules

Example
-------
python scripts/regression/plot_beta_summary.py \
    --inputs beta_0 beta_0.5 beta_1 beta_2 \
    --output-name figure8_beta_summary

Absolute paths and paths relative to the current working directory are also
accepted.

Outputs
-------
results/figures/regression/figure8_beta_summary.png
results/figures/regression/figure8_beta_summary.pdf
results/figures/regression/figure8_beta_summary.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BENCHMARK_ROOT = Path("results/regression/benchmarks")
DEFAULT_OUTPUT_DIR = Path("results/figures/regression")

DEFAULT_METRICS = (
    "rmse",
    "pearson",
    "best_discovered",
    "top20_mean_discovered",
)

METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": r"$R^2$",
    "pearson": "Pearson correlation",
    "best_discovered": "Best affinity discovered",
    "top20_mean_discovered": "Mean top-20 affinity",
    "mean_discovered": "Mean discovered affinity",
}

MODEL_LABELS = {
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "bayesian_ridge": "Bayesian ridge",
    "gaussian_process": "Gaussian process",
    "gradient_boosting": "Gradient boosting",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "knn": "KNN",
    "linear_regression": "Linear regression",
}

MODEL_ORDER = (
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
    "gradient_boosting",
    "hist_gradient_boosting",
    "knn",
    "linear_regression",
)

# Fixed model colours keep regression figures visually consistent.
MODEL_COLORS = {
    "random_forest": "#4C72B0",
    "extra_trees": "#55A868",
    "bayesian_ridge": "#C44E52",
    "gaussian_process": "#8172B2",
    "gradient_boosting": "#CCB974",
    "hist_gradient_boosting": "#64B5CD",
    "knn": "#8C8C8C",
    "linear_regression": "#E17C05",
}

MARKERS = (
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "<",
)


@dataclass(frozen=True)
class BenchmarkData:
    """Data loaded from one benchmark subdirectory."""

    path: Path
    beta: float
    target: str | None
    strategy: str | None
    summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare regression benchmark results across UCB beta values."
        )
    )
    parser.add_argument(
        "--inputs",
        "--input",
        dest="inputs",
        nargs="+",
        required=True,
        metavar="SUBPATH",
        help=(
            "Benchmark subdirectories. Relative paths are first resolved "
            "inside --benchmark-root. Example: "
            "--inputs beta_0 beta_0.5 beta_1 beta_2"
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help=(
            "Base directory containing benchmark subdirectories "
            f"(default: {DEFAULT_BENCHMARK_ROOT})."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs=4,
        default=list(DEFAULT_METRICS),
        metavar=("METRIC_A", "METRIC_B", "METRIC_C", "METRIC_D"),
        help=(
            "Exactly four metric prefixes to plot. The script expects "
            "<metric>_mean and <metric>_std columns in "
            "final_round_summary.csv. Defaults: "
            + " ".join(DEFAULT_METRICS)
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional subset and order of models. By default, all models "
            "shared by every input benchmark are plotted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory in which PNG, PDF and SVG files are saved "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="beta_summary_panel",
        help="Output filename without extension.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure-level title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster resolution for the PNG output (default: 300).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(11.5, 8.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 11.5 8.5).",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=4,
        help="Number of columns in the shared legend (default: 4).",
    )
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Do not draw ±1 standard-deviation bands.",
    )
    return parser.parse_args()


def resolve_input_path(raw_path: str, benchmark_root: Path) -> Path:
    """
    Resolve a user-supplied benchmark path.

    Resolution order:
    1. absolute path;
    2. path relative to the current working directory;
    3. path relative to benchmark_root.
    """
    supplied = Path(raw_path).expanduser()

    if supplied.is_absolute():
        candidate = supplied
    elif supplied.exists():
        candidate = supplied
    else:
        candidate = benchmark_root / supplied

    candidate = candidate.resolve()

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Benchmark directory not found: {candidate}"
        )

    return candidate


def load_benchmark(path: Path) -> BenchmarkData:
    config_path = path / "config.json"
    summary_path = path / "final_round_summary.csv"

    missing = [
        file_path.name
        for file_path in (config_path, summary_path)
        if not file_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{path} is missing required file(s): {', '.join(missing)}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if "beta" not in config:
        raise KeyError(
            f"'beta' is missing from {config_path}"
        )

    beta = float(config["beta"])
    summary = pd.read_csv(summary_path)

    if "model" not in summary.columns:
        raise ValueError(
            f"'model' column is missing from {summary_path}"
        )

    if summary["model"].duplicated().any():
        duplicated = (
            summary.loc[summary["model"].duplicated(), "model"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicate model rows in {summary_path}: {duplicated}"
        )

    return BenchmarkData(
        path=path,
        beta=beta,
        target=config.get("target"),
        strategy=config.get("strategy"),
        summary=summary.copy(),
    )


def validate_benchmarks(
    benchmarks: list[BenchmarkData],
    metrics: Iterable[str],
) -> None:
    if not benchmarks:
        raise ValueError("No benchmark inputs were provided.")

    beta_values = [benchmark.beta for benchmark in benchmarks]
    if len(beta_values) != len(set(beta_values)):
        raise ValueError(
            "Each input must represent a unique beta value. "
            f"Received: {beta_values}"
        )

    targets = {
        benchmark.target
        for benchmark in benchmarks
        if benchmark.target is not None
    }
    if len(targets) > 1:
        raise ValueError(
            "Inputs contain different targets: "
            + ", ".join(sorted(targets))
        )

    strategies = {
        benchmark.strategy
        for benchmark in benchmarks
        if benchmark.strategy is not None
    }
    if len(strategies) > 1:
        raise ValueError(
            "Inputs contain different strategies: "
            + ", ".join(sorted(strategies))
        )

    required_columns = {
        column
        for metric in metrics
        for column in (f"{metric}_mean", f"{metric}_std")
    }

    for benchmark in benchmarks:
        missing = required_columns.difference(
            benchmark.summary.columns
        )
        if missing:
            raise ValueError(
                f"{benchmark.path / 'final_round_summary.csv'} "
                "is missing metric columns: "
                + ", ".join(sorted(missing))
            )


def select_models(
    benchmarks: list[BenchmarkData],
    requested_models: list[str] | None,
) -> list[str]:
    model_sets = [
        set(benchmark.summary["model"].astype(str))
        for benchmark in benchmarks
    ]
    shared_models = set.intersection(*model_sets)

    if not shared_models:
        raise ValueError(
            "The input benchmarks do not share any model names."
        )

    union_models = set.union(*model_sets)
    if shared_models != union_models:
        missing_report = []
        for benchmark in benchmarks:
            available = set(
                benchmark.summary["model"].astype(str)
            )
            absent = sorted(union_models - available)
            if absent:
                missing_report.append(
                    f"{benchmark.path.name}: {', '.join(absent)}"
                )
        print(
            "Warning: some models are not available in every benchmark. "
            "Only shared models will be plotted.\n  "
            + "\n  ".join(missing_report),
            file=sys.stderr,
        )

    if requested_models is not None:
        missing_requested = [
            model
            for model in requested_models
            if model not in shared_models
        ]
        if missing_requested:
            raise ValueError(
                "Requested model(s) are not available in every input: "
                + ", ".join(missing_requested)
            )
        return requested_models

    ordered = [
        model
        for model in MODEL_ORDER
        if model in shared_models
    ]
    unknown = sorted(shared_models.difference(ordered))

    return ordered + unknown


def combine_summaries(
    benchmarks: list[BenchmarkData],
    models: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for benchmark in benchmarks:
        frame = benchmark.summary.copy()
        frame["model"] = frame["model"].astype(str)
        frame = frame.loc[frame["model"].isin(models)].copy()
        frame["beta"] = benchmark.beta
        frame["benchmark_path"] = str(benchmark.path)
        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined["model"] = pd.Categorical(
        combined["model"],
        categories=models,
        ordered=True,
    )

    return combined.sort_values(
        ["model", "beta"]
    ).reset_index(drop=True)


def format_beta(beta: float) -> str:
    """Format beta labels without unnecessary trailing zeroes."""
    if float(beta).is_integer():
        return str(int(beta))
    return f"{beta:g}"


def model_label(model: str) -> str:
    return MODEL_LABELS.get(
        model,
        model.replace("_", " ").title(),
    )


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def style_axis(
    ax: plt.Axes,
    beta_values: np.ndarray,
    metric: str,
    panel_letter: str,
) -> None:
    ax.set_xlabel(r"Exploration parameter, $\beta$")
    ax.set_ylabel(metric_label(metric))

    ax.set_xticks(beta_values)
    ax.set_xticklabels(
        [format_beta(beta) for beta in beta_values]
    )

    ax.grid(
        axis="both",
        linewidth=0.6,
        alpha=0.25,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        -0.13,
        1.07,
        panel_letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_panel(
    data: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    *,
    figsize: tuple[float, float],
    title: str | None,
    legend_columns: int,
    show_uncertainty: bool,
) -> plt.Figure:
    beta_values = np.array(
        sorted(data["beta"].astype(float).unique()),
        dtype=float,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex=True,
    )
    axes_flat = axes.ravel()
    panel_letters = "ABCD"

    fallback_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 1.0, max(len(models), 2))
    )

    for metric_index, (ax, metric) in enumerate(
        zip(axes_flat, metrics, strict=True)
    ):
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        for model_index, model in enumerate(models):
            model_data = (
                data.loc[data["model"] == model]
                .sort_values("beta")
            )

            x = model_data["beta"].to_numpy(dtype=float)
            mean = model_data[mean_column].to_numpy(dtype=float)
            std = model_data[std_column].to_numpy(dtype=float)

            color = MODEL_COLORS.get(
                model,
                fallback_colors[model_index],
            )
            marker = MARKERS[model_index % len(MARKERS)]

            ax.plot(
                x,
                mean,
                label=model_label(model),
                color=color,
                marker=marker,
                markersize=5.5,
                linewidth=1.8,
                markeredgewidth=0.7,
                zorder=3,
            )

            if show_uncertainty:
                valid = (
                    np.isfinite(x)
                    & np.isfinite(mean)
                    & np.isfinite(std)
                )
                if valid.any():
                    ax.fill_between(
                        x[valid],
                        mean[valid] - std[valid],
                        mean[valid] + std[valid],
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                        zorder=1,
                    )

        style_axis(
            ax,
            beta_values,
            metric,
            panel_letters[metric_index],
        )

    # Obtain one shared legend from the first axis.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(legend_columns, len(models)),
        frameon=False,
        columnspacing=1.5,
        handlelength=2.2,
        fontsize=9,
    )

    if title:
        fig.suptitle(
            title,
            fontsize=14,
            y=0.995,
        )

    # Manual spacing is more predictable than bbox-only layout when a
    # shared legend sits below the axes.
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.93 if title else 0.96,
        bottom=0.17,
        wspace=0.28,
        hspace=0.30,
    )

    return fig


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    output_name: str,
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths: list[Path] = []

    for extension in ("png", "pdf", "svg"):
        output_path = output_dir / f"{output_name}.{extension}"

        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
        }
        if extension == "png":
            save_kwargs["dpi"] = dpi

        fig.savefig(
            output_path,
            **save_kwargs,
        )
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    args = parse_args()

    try:
        benchmark_paths = [
            resolve_input_path(
                raw_path,
                args.benchmark_root,
            )
            for raw_path in args.inputs
        ]

        benchmarks = [
            load_benchmark(path)
            for path in benchmark_paths
        ]

        validate_benchmarks(
            benchmarks,
            args.metrics,
        )

        models = select_models(
            benchmarks,
            args.models,
        )

        data = combine_summaries(
            benchmarks,
            models,
        )

        fig = plot_panel(
            data,
            models,
            args.metrics,
            figsize=tuple(args.figsize),
            title=args.title,
            legend_columns=args.legend_columns,
            show_uncertainty=not args.no_uncertainty,
        )

        saved_paths = save_figure(
            fig,
            args.output_dir,
            args.output_name,
            args.dpi,
        )
        plt.close(fig)

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ) as error:
        raise SystemExit(
            f"Error: {error}"
        ) from error

    print("Compared beta values:")
    for benchmark in sorted(
        benchmarks,
        key=lambda item: item.beta,
    ):
        print(
            f"  beta={format_beta(benchmark.beta)} "
            f"-> {benchmark.path}"
        )

    print("\nModels:")
    for model in models:
        print(f"  {model}")

    print("\nSaved:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
