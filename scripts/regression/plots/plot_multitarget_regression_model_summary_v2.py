#!/usr/bin/env python3
"""
Figure 9: Multi-target regression surrogate-model benchmark.

Scientific question
-------------------
Does the surrogate model that predicts affinity best also make the best
active-learning decisions?

The figure deliberately separates:
    A-C  Full 8-model benchmark
    D-H  Uncertainty-aware UCB subset

Panels
------
A  Final-round RMSE heatmap across 10 targets × 8 models
B  Final-round Top-20 discovered affinity heatmap
C  Mean prediction rank vs mean discovery rank for all 8 models
D  Prediction ranks for uncertainty-aware models
E  Discovery ranks for uncertainty-aware models
F  RF - GP Top-20 discovery difference by target, paired by seed
G  Prediction rank across active-learning rounds, uncertainty-aware models
H  Discovery rank across active-learning rounds, uncertainty-aware models

Inputs
------
results/regression/analysis/multitarget/
    canonical_history.csv
    canonical_round_summary.csv
    final_model_target_summary.csv
    model_ranks_all.csv
    model_ranks_uncertainty_aware.csv

Outputs
-------
results/figures/regression/
    figure9_multitarget_regression_model_summary.png
    figure9_multitarget_regression_model_summary.pdf
    figure9_multitarget_regression_model_summary.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_ROOT = Path("results/regression/analysis/multitarget")
DEFAULT_OUTPUT_DIR = Path("results/figures/regression")
DEFAULT_OUTPUT_NAME = "figure9_multitarget_regression_model_summary"

TARGETS = [
    "EGFR",
    "JAK2",
    "PARP1",
    "BRAF",
    "ABL1",
    "SRC",
    "VEGFR2",
    "CDK2",
    "DRD2",
    "CA2",
]

MODELS = [
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
    "gradient_boosting",
    "hist_gradient_boosting",
    "knn",
    "linear_regression",
]

UQ_MODELS = [
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
]

MODEL_LABELS = {
    "random_forest": "RF",
    "extra_trees": "ET",
    "bayesian_ridge": "BR",
    "gaussian_process": "GP",
    "gradient_boosting": "GB",
    "hist_gradient_boosting": "HGB",
    "knn": "KNN",
    "linear_regression": "LR",
}

MODEL_LONG_LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "bayesian_ridge": "Bayesian Ridge",
    "gaussian_process": "Gaussian Process",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "knn": "K-Nearest Neighbours",
    "linear_regression": "Linear Regression",
}

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

MODEL_MARKERS = {
    "random_forest": "o",
    "extra_trees": "s",
    "bayesian_ridge": "^",
    "gaussian_process": "D",
    "gradient_boosting": "v",
    "hist_gradient_boosting": "P",
    "knn": "X",
    "linear_regression": "<",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 9 from the canonical multi-target regression analysis."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(17.5, 14.0),
        metavar=("WIDTH", "HEIGHT"),
    )
    return parser.parse_args()


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    required = {
        "history": "canonical_history.csv",
        "rounds": "canonical_round_summary.csv",
        "final": "final_model_target_summary.csv",
        "ranks_all": "model_ranks_all.csv",
        "ranks_uq": "model_ranks_uncertainty_aware.csv",
    }

    data: dict[str, pd.DataFrame] = {}
    for key, filename in required.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
        data[key] = pd.read_csv(path)
    return data


def validate_inputs(data: dict[str, pd.DataFrame]) -> None:
    final = data["final"]
    history = data["history"]
    rounds = data["rounds"]

    checks = [
        (final, ["target", "model", "rmse_mean", "top20_mean_discovered_mean"]),
        (history, ["target", "model", "seed", "round", "top20_mean_discovered"]),
        (rounds, ["target", "model", "round", "rmse_mean", "top20_mean_discovered_mean"]),
    ]
    for df, required in checks:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    expected = {(t, m) for t in TARGETS for m in MODELS}
    observed = set(zip(final["target"], final["model"]))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing target/model pairs: {missing}")


def model_rank_summary(ranks: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    subset = ranks[ranks["model"].isin(models)].copy()
    return (
        subset.groupby("model", as_index=False)
        .agg(
            prediction_mean=("prediction_composite_rank", "mean"),
            prediction_std=("prediction_composite_rank", "std"),
            discovery_mean=("discovery_composite_rank", "mean"),
            discovery_std=("discovery_composite_rank", "std"),
        )
        .set_index("model")
        .reindex(models)
        .reset_index()
    )


def heatmap_matrix(final: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        final.pivot(index="target", columns="model", values=metric)
        .reindex(index=TARGETS, columns=MODELS)
        .to_numpy(dtype=float)
    )


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, fmt: str) -> None:
    finite = values[np.isfinite(values)]
    midpoint = (np.nanmin(finite) + np.nanmax(finite)) / 2.0 if finite.size else 0.0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "NA" if not np.isfinite(value) else format(value, fmt)
            color = "white" if np.isfinite(value) and value < midpoint else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7.0, color=color, fontweight="bold")


def plot_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    fmt: str,
    cmap: str,
) -> None:
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(TARGETS)))
    ax.set_yticklabels(TARGETS, fontsize=8)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    annotate_heatmap(ax, values, fmt)

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.axvline(3.5, color="white", linewidth=2.5)
    ax.text(1.5, -1.35, "Uncertainty-aware", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(5.5, -1.35, "Deterministic", ha="center", va="center",
            fontsize=8, fontweight="bold")


def plot_prediction_discovery_scatter(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        model = row["model"]
        x = float(row["prediction_mean"])
        y = float(row["discovery_mean"])
        ax.scatter(x, y, s=70, color=MODEL_COLORS[model],
                   marker=MODEL_MARKERS[model], edgecolor="black",
                   linewidth=0.5, zorder=3)
        ax.annotate(MODEL_LABELS[model], (x, y), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, fontweight="bold")

    ax.plot([1, len(MODELS)], [1, len(MODELS)],
            linestyle="--", linewidth=1.0, color="0.55", zorder=1)
    ax.set_xlim(0.7, len(MODELS) + 0.3)
    ax.set_ylim(0.7, len(MODELS) + 0.3)
    ax.set_xlabel("Mean prediction rank\n(lower is better)", fontsize=8.5)
    ax.set_ylabel("Mean discovery rank\n(lower is better)", fontsize=8.5)
    ax.set_title("Prediction and discovery are distinct objectives",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_rank_bars(ax: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    frame = summary.sort_values(mean_col).copy()
    models = frame["model"].tolist()
    y = np.arange(len(models))

    ax.barh(
        y,
        frame[mean_col],
        xerr=frame[std_col],
        color=[MODEL_COLORS[m] for m in models],
        alpha=0.88,
        edgecolor="none",
        error_kw={"elinewidth": 1.0, "capsize": 2.5},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LONG_LABELS[m] for m in models], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank across targets\n(lower is better)", fontsize=8.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlim(0.7, len(UQ_MODELS) + 0.4)
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def paired_rf_gp_discovery(history: pd.DataFrame) -> pd.DataFrame:
    subset = history[history["model"].isin(["random_forest", "gaussian_process"])].copy()
    final_round = int(subset["round"].max())
    subset = subset[subset["round"] == final_round].copy()

    pivot = subset.pivot_table(
        index=["target", "seed"],
        columns="model",
        values="top20_mean_discovered",
        aggfunc="first",
    )
    pivot["difference"] = pivot["random_forest"] - pivot["gaussian_process"]

    return (
        pivot.reset_index()
        .groupby("target", as_index=False)
        .agg(
            mean_difference=("difference", "mean"),
            std_difference=("difference", "std"),
            n_seeds=("difference", "count"),
        )
        .set_index("target")
        .reindex(TARGETS)
        .reset_index()
    )


def plot_rf_gp_difference(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary.sort_values("mean_difference").copy()
    y = np.arange(len(frame))
    ax.barh(
        y,
        frame["mean_difference"],
        xerr=frame["std_difference"],
        color=MODEL_COLORS["random_forest"],
        alpha=0.85,
        edgecolor="none",
        error_kw={"elinewidth": 1.0, "capsize": 2.5},
    )
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["target"], fontsize=8)
    ax.set_xlabel(r"$\Delta$Top-20 affinity (RF - GP)", fontsize=8.5)
    ax.set_title("Random Forest discovery advantage by target",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def round_ranks(rounds: pd.DataFrame) -> pd.DataFrame:
    subset = rounds[rounds["model"].isin(UQ_MODELS)].copy()
    subset["prediction_rank"] = subset.groupby(["target", "round"])["rmse_mean"].rank(
        method="average", ascending=True
    )
    subset["discovery_rank"] = subset.groupby(["target", "round"])["top20_mean_discovered_mean"].rank(
        method="average", ascending=False
    )
    return (
        subset.groupby(["round", "model"], as_index=False)
        .agg(
            prediction_mean=("prediction_rank", "mean"),
            prediction_std=("prediction_rank", "std"),
            discovery_mean=("discovery_rank", "mean"),
            discovery_std=("discovery_rank", "std"),
        )
    )


def plot_rank_curves(ax: plt.Axes, rank_rounds: pd.DataFrame, metric: str, title: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for model in UQ_MODELS:
        frame = rank_rounds[rank_rounds["model"] == model].sort_values("round")
        x = frame["round"].to_numpy(float)
        mean = frame[mean_col].to_numpy(float)
        std = frame[std_col].to_numpy(float)

        ax.plot(
            x, mean,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markevery=max(1, len(x)//10),
            markersize=4,
            linewidth=1.7,
            label=MODEL_LONG_LABELS[model],
        )
        ax.fill_between(
            x, mean-std, mean+std,
            color=MODEL_COLORS[model],
            alpha=0.10,
            linewidth=0,
        )

    ax.set_xlabel("Active-learning round", fontsize=8.5)
    ax.set_ylabel("Mean rank across targets\n(lower is better)", fontsize=8.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_ylim(len(UQ_MODELS) + 0.35, 0.65)
    ax.set_yticks(np.arange(1, len(UQ_MODELS) + 1))
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12, 1.08, letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )


def create_figure(data: dict[str, pd.DataFrame], figsize: tuple[float, float]) -> plt.Figure:
    final = data["final"]
    history = data["history"]
    rounds = data["rounds"]
    ranks_all = data["ranks_all"]
    ranks_uq = data["ranks_uq"]

    rmse_matrix = heatmap_matrix(final, "rmse_mean")
    top20_matrix = heatmap_matrix(final, "top20_mean_discovered_mean")
    all_rank_summary = model_rank_summary(ranks_all, MODELS)
    uq_rank_summary = model_rank_summary(ranks_uq, UQ_MODELS)
    rf_gp = paired_rf_gp_discovery(history)
    rank_rounds = round_ranks(rounds)

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        3, 6,
        height_ratios=[1.0, 0.86, 0.90],
        hspace=0.48,
        wspace=0.48,
    )

    axes = [
        fig.add_subplot(outer[0, 0:2]),
        fig.add_subplot(outer[0, 2:4]),
        fig.add_subplot(outer[0, 4:6]),
        fig.add_subplot(outer[1, 0:2]),
        fig.add_subplot(outer[1, 2:4]),
        fig.add_subplot(outer[1, 4:6]),
        fig.add_subplot(outer[2, 0:3]),
        fig.add_subplot(outer[2, 3:6]),
    ]

    plot_heatmap(axes[0], rmse_matrix, "Final predictive performance", "RMSE", ".2f", "viridis_r")
    plot_heatmap(axes[1], top20_matrix, "Final lead-discovery performance",
                 "Top-20 mean affinity", ".2f", "viridis")
    plot_prediction_discovery_scatter(axes[2], all_rank_summary)
    plot_rank_bars(axes[3], uq_rank_summary, "prediction",
                   "Prediction ranking: uncertainty-aware models")
    plot_rank_bars(axes[4], uq_rank_summary, "discovery",
                   "Discovery ranking: uncertainty-aware models")
    plot_rf_gp_difference(axes[5], rf_gp)
    plot_rank_curves(axes[6], rank_rounds, "prediction",
                     "Predictive ranking across acquisition rounds")
    plot_rank_curves(axes[7], rank_rounds, "discovery",
                     "Discovery ranking across acquisition rounds")

    for letter, ax in zip("ABCDEFGH", axes):
        add_panel_letter(ax, letter)

    fig.text(0.5, 0.955, "All regression models",
             ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.635, "Uncertainty-aware UCB models (β = 2)",
             ha="center", va="center", fontsize=11, fontweight="bold")

    handles = [
        plt.Line2D(
            [0], [0],
            color=MODEL_COLORS[m],
            marker=MODEL_MARKERS[m],
            linewidth=1.7,
            markersize=5,
            label=MODEL_LONG_LABELS[m],
        )
        for m in UQ_MODELS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.8,
        handlelength=2.4,
    )

    fig.suptitle(
        "Multi-target regression benchmark: prediction and molecular discovery",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.075)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, output_name: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{output_name}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    return saved


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Analysis directory not found: {root}")

    try:
        data = load_inputs(root)
        validate_inputs(data)
        fig = create_figure(data, tuple(args.figsize))
        saved = save_figure(fig, args.output_dir, args.output_name, args.dpi)
        plt.close(fig)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print("Saved Figure 9:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
