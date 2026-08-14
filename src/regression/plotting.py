"""Reusable plotting utilities for regression active-learning benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    label: str
    filename: str
    higher_is_better: bool


METRICS: Mapping[str, MetricSpec] = {
    "rmse": MetricSpec("RMSE", "rmse_vs_round.png", False),
    "mae": MetricSpec("MAE", "mae_vs_round.png", False),
    "r2": MetricSpec(r"$R^2$", "r2_vs_round.png", True),
    "pearson": MetricSpec("Pearson correlation", "pearson_vs_round.png", True),
    "best_discovered": MetricSpec(
        "Best discovered affinity",
        "best_discovered_vs_round.png",
        True,
    ),
    "top20_mean_discovered": MetricSpec(
        "Mean affinity of top 20 discovered",
        "top20_mean_discovered_vs_round.png",
        True,
    ),
    "mean_discovered": MetricSpec(
        "Mean discovered affinity",
        "mean_discovered_vs_round.png",
        True,
    ),
    "fraction_best_found": MetricSpec(
        "Fraction of dataset optimum recovered",
        "fraction_best_found_vs_round.png",
        True,
    ),
    "distance_to_best": MetricSpec(
        "Distance to dataset optimum",
        "distance_to_best_vs_round.png",
        False,
    ),
    "pool_mean_uncertainty": MetricSpec(
        "Mean pool uncertainty",
        "pool_mean_uncertainty_vs_round.png",
        False,
    ),
    "pool_max_uncertainty": MetricSpec(
        "Maximum pool uncertainty",
        "pool_max_uncertainty_vs_round.png",
        False,
    ),
    "round_runtime_seconds": MetricSpec(
        "Round runtime (s)",
        "round_runtime_vs_round.png",
        False,
    ),
}


MODEL_LABELS: Mapping[str, str] = {
    "bayesian_ridge": "Bayesian ridge",
    "extra_trees": "Extra trees",
    "gaussian_process": "Gaussian process",
    "gradient_boosting": "Gradient boosting",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "knn": "KNN",
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
}


MODEL_COLORS: Mapping[str, str] = {
    "bayesian_ridge": "#1f77b4",
    "extra_trees": "#ff7f0e",
    "gaussian_process": "#2ca02c",
    "gradient_boosting": "#d62728",
    "hist_gradient_boosting": "#9467bd",
    "knn": "#8c564b",
    "linear_regression": "#e377c2",
    "random_forest": "#7f7f7f",
}


def set_plot_style() -> None:
    """Apply one consistent, publication-oriented matplotlib style."""
    plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
        }
    )


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_csv(
    path: str | Path,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = pd.read_csv(csv_path)

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: "
            f"{', '.join(missing)}"
        )

    return frame


def load_round_summary(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ["model", "round", "beta"])


def load_final_round_summary(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ["model"])


def save_figure(
    fig: plt.Figure,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    ensure_output_dir(output_path.parent)

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"✓ {output_path}")

    return output_path


def _metric_columns(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[str, str | None]:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"

    if mean_column not in frame.columns:
        raise ValueError(
            f"Metric '{metric}' is unavailable: "
            f"expected column '{mean_column}'."
        )

    if std_column not in frame.columns:
        std_column = None

    return mean_column, std_column


def model_label(model: str) -> str:
    return MODEL_LABELS.get(
        model,
        model.replace("_", " ").title(),
    )


def model_color(model: str) -> str | None:
    return MODEL_COLORS.get(model)


def _selected_models(
    frame: pd.DataFrame,
    models: Iterable[str] | None,
) -> list[str]:
    if models is not None:
        return list(models)

    available = set(frame["model"].unique())

    preferred_order = [
        model
        for model in MODEL_LABELS
        if model in available
    ]

    remaining = sorted(
        available - set(preferred_order)
    )

    return preferred_order + remaining


def _place_legend_below(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ncol: int = 4,
) -> None:
    ax.legend(
        frameon=False,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        borderaxespad=0.0,
    )

    fig.subplots_adjust(bottom=0.30)


def _plot_metric_on_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    models: Iterable[str] | None = None,
    show_std: bool = True,
    drop_last_round: bool = False,
) -> None:
    if metric not in METRICS:
        raise KeyError(
            f"Unknown metric '{metric}'. "
            "Add it to METRICS first."
        )

    mean_column, std_column = _metric_columns(
        frame,
        metric,
    )

    selected_models = _selected_models(
        frame,
        models,
    )

    plot_frame = frame.copy()

    if drop_last_round:
        last_round = plot_frame["round"].max()

        plot_frame = plot_frame.loc[
            plot_frame["round"] < last_round
        ]

    for model in selected_models:
        model_frame = plot_frame.loc[
            plot_frame["model"] == model
        ].sort_values("round")

        if model_frame.empty:
            raise ValueError(
                f"Model '{model}' is absent "
                "from the round summary."
            )

        rounds = model_frame[
            "round"
        ].to_numpy()

        means = model_frame[
            mean_column
        ].to_numpy(dtype=float)

        color = model_color(model)

        (line,) = ax.plot(
            rounds,
            means,
            label=model_label(model),
            color=color,
        )

        if show_std and std_column is not None:
            stds = model_frame[
                std_column
            ].to_numpy(dtype=float)

            ax.fill_between(
                rounds,
                means - stds,
                means + stds,
                color=line.get_color(),
                alpha=0.13,
                linewidth=0,
            )

    spec = METRICS[metric]

    ax.set_xlabel("Active-learning round")
    ax.set_ylabel(spec.label)
    ax.set_title(title or spec.label)


def plot_metric_vs_round(
    frame: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    title: str | None = None,
    models: Iterable[str] | None = None,
    show_std: bool = True,
    drop_last_round: bool = False,
) -> Path:
    """Plot one round-level metric for all requested models."""
    fig, ax = plt.subplots()

    _plot_metric_on_axis(
        ax,
        frame,
        metric,
        title=title,
        models=models,
        show_std=show_std,
        drop_last_round=drop_last_round,
    )

    _place_legend_below(
        fig,
        ax,
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_metric_panel(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    output_path: str | Path,
    *,
    ncols: int = 2,
    show_std: bool = True,
    drop_last_round_for: Iterable[str] = (),
) -> Path:
    """Create a multi-panel summary figure."""
    if not metrics:
        raise ValueError(
            "At least one metric is required."
        )

    nrows = int(
        np.ceil(len(metrics) / ncols)
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.2 * ncols, 4.8 * nrows),
        squeeze=False,
    )

    flat_axes = axes.ravel()

    drop_metrics = set(
        drop_last_round_for
    )

    for ax, metric in zip(
        flat_axes,
        metrics,
    ):
        _plot_metric_on_axis(
            ax,
            frame,
            metric,
            show_std=show_std,
            drop_last_round=metric in drop_metrics,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for ax in flat_axes[len(metrics):]:
        ax.set_visible(False)

    handles, labels = (
        flat_axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.subplots_adjust(
        bottom=0.13,
        hspace=0.30,
        wspace=0.23,
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_final_metric_ranking(
    frame: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot final-round model performance as a forest plot."""
    if metric not in METRICS:
        raise KeyError(
            f"Unknown metric '{metric}'. "
            "Add it to METRICS first."
        )

    mean_column, std_column = _metric_columns(
        frame,
        metric,
    )

    spec = METRICS[metric]

    ranked = frame.copy()

    ranked = ranked.sort_values(
        mean_column,
        ascending=not spec.higher_is_better,
    )

    labels = [
        model_label(model)
        for model in ranked["model"]
    ]

    means = ranked[
        mean_column
    ].to_numpy(dtype=float)

    if std_column is not None:
        errors = ranked[
            std_column
        ].to_numpy(dtype=float)
    else:
        errors = np.zeros_like(means)

    y_positions = np.arange(
        len(ranked)
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0)
    )

    for y, model, mean, error in zip(
        y_positions,
        ranked["model"],
        means,
        errors,
    ):
        ax.errorbar(
            mean,
            y,
            xerr=error if std_column is not None else None,
            fmt="o",
            markersize=7,
            capsize=4,
            elinewidth=1.6,
            color=model_color(model),
        )

    ax.set_yticks(
        y_positions,
        labels,
    )

    ax.invert_yaxis()

    ax.set_xlabel(spec.label)

    ax.set_title(
        title
        or f"Final-round {spec.label.lower()}"
    )

    if metric == "r2":
        ax.axvline(
            0.0,
            color="0.45",
            linestyle="--",
            linewidth=1.0,
        )

    fig.tight_layout()

    return save_figure(
        fig,
        output_path,
    )


def plot_beta_comparison(
    frames: Mapping[float, pd.DataFrame],
    metric: str,
    output_path: str | Path,
    *,
    models: Iterable[str] | None = None,
    title: str | None = None,
    show_std: bool = False,
) -> Path:
    """Compare one metric across multiple beta values."""
    if metric not in METRICS:
        raise KeyError(
            f"Unknown metric '{metric}'. "
            "Add it to METRICS first."
        )

    if len(frames) < 2:
        raise ValueError(
            "At least two beta datasets are required."
        )

    first_frame = next(
        iter(frames.values())
    )

    selected_models = _selected_models(
        first_frame,
        models,
    )

    common_models = set.intersection(
        *(
            set(frame["model"])
            for frame in frames.values()
        )
    )

    selected_models = [
        model
        for model in selected_models
        if model in common_models
    ]

    line_styles = [
        "-",
        "--",
        ":",
        "-.",
    ]

    fig, ax = plt.subplots(
        figsize=(8.4, 5.2)
    )

    for beta_index, (beta, frame) in enumerate(
        sorted(frames.items())
    ):
        mean_column, std_column = _metric_columns(
            frame,
            metric,
        )

        for model in selected_models:
            model_frame = frame.loc[
                frame["model"] == model
            ].sort_values("round")

            if model_frame.empty:
                raise ValueError(
                    f"Model '{model}' is absent "
                    f"for beta={beta}."
                )

            rounds = model_frame[
                "round"
            ].to_numpy()

            means = model_frame[
                mean_column
            ].to_numpy(dtype=float)

            (line,) = ax.plot(
                rounds,
                means,
                linestyle=line_styles[
                    beta_index % len(line_styles)
                ],
                color=model_color(model),
                label=(
                    f"{model_label(model)} "
                    f"— β={beta:g}"
                ),
            )

            if show_std and std_column is not None:
                stds = model_frame[
                    std_column
                ].to_numpy(dtype=float)

                ax.fill_between(
                    rounds,
                    means - stds,
                    means + stds,
                    color=line.get_color(),
                    alpha=0.07,
                    linewidth=0,
                )

    spec = METRICS[metric]

    ax.set_xlabel("Active-learning round")
    ax.set_ylabel(spec.label)

    ax.set_title(
        title
        or (
            f"{spec.label}: "
            "exploration-weight comparison"
        )
    )

    _place_legend_below(
        fig,
        ax,
        ncol=4,
    )

    return save_figure(
        fig,
        output_path,
    )