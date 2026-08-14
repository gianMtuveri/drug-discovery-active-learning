"""Generate the minimal figure set for Part I of the active-learning article."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from src.toy_data import make_toy_dataset

HISTORY_PATH = Path("results/tables/toy_initialization_comparison_history.csv")
OUTPUT_DIR = Path("results/figures")
STRATEGIES = ["random", "greedy", "uncertainty_topk"]
DISPLAY_NAMES = {
    "random": "Random",
    "greedy": "Greedy",
    "uncertainty_topk": "Uncertainty",
}
COLORS = {
    "random": "#4C78A8",
    "greedy": "#E45756",
    "uncertainty_topk": "#54A24B",
}
LINESTYLES = {
    "random": "-",
    "greedy": "--",
    "uncertainty_topk": "-.",
}


def parse_indices(value: object) -> np.ndarray:
    """Convert a saved index collection into a one-dimensional integer array."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=int)
    if isinstance(value, np.ndarray):
        return value.astype(int, copy=False).ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=int).ravel()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return np.array([], dtype=int)

    try:
        return np.asarray(ast.literal_eval(text), dtype=int).ravel()
    except (SyntaxError, ValueError):
        clean = text.strip("[]").replace(",", " ")
        return np.fromstring(clean, sep=" ", dtype=int)


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"History file not found: {path}\n"
            "Run scripts/run_toy_repeated_simulation.py first."
        )
    history = pd.read_csv(path)
    required = {
        "round", "strategy", "seed", "initialization_strategy", "roc_auc",
        "initial_indices", "labeled_indices", "selected_indices",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError("History is missing columns: " + ", ".join(sorted(missing)))
    return history


def make_pool() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_toy_dataset(n_samples=1000, random_state=42)
    X_pool, _, y_pool, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_pool, y_pool


def configure_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
            "svg.fonttype": "none",
    })


def choose_representative_seed(
    history: pd.DataFrame,
    initialization: str,
    strategies: Iterable[str],
    target_round: int,
) -> int:
    subset = history[
        (history["initialization_strategy"] == initialization)
        & (history["strategy"].isin(strategies))
        & (history["round"] == target_round)
    ].copy()
    medians = subset.groupby("strategy")["roc_auc"].median()
    subset["absolute_deviation"] = subset.apply(
        lambda row: abs(row["roc_auc"] - medians.loc[row["strategy"]]), axis=1
    )
    seed_scores = subset.groupby("seed")["absolute_deviation"].mean()
    if seed_scores.empty:
        raise ValueError("No complete final-round campaigns were found.")
    return int(seed_scores.idxmin())


def plot_acquisition_behaviour(
    history: pd.DataFrame,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    output_path: Path,
    initialization: str = "random",
    target_round: int = 10,
) -> int:
    seed = choose_representative_seed(
        history, initialization, STRATEGIES, target_round
    )
    fig, axes = plt.subplots(
        1, len(STRATEGIES), figsize=(10.5, 3.45), sharex=True, sharey=True,
        constrained_layout=True,
    )

    for ax, strategy in zip(axes, STRATEGIES):
        rows = history[
            (history["initialization_strategy"] == initialization)
            & (history["strategy"] == strategy)
            & (history["seed"] == seed)
            & (history["round"] == target_round)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one row for {strategy}, seed={seed}, round={target_round}; "
                f"found {len(rows)}."
            )
        row = rows.iloc[0]
        initial = parse_indices(row["initial_indices"])
        labelled = parse_indices(row["labeled_indices"])
        acquired = np.setdiff1d(labelled, initial, assume_unique=False)

        ax.scatter(
            X_pool[:, 0], X_pool[:, 1], s=8, c="#D9D9D9", alpha=0.45,
            linewidths=0, rasterized=True,
        )
        inactive = acquired[y_pool[acquired] == 0]
        active = acquired[y_pool[acquired] == 1]
        ax.scatter(
            X_pool[inactive, 0], X_pool[inactive, 1], s=21,
            facecolors="none", edgecolors=COLORS[strategy], linewidths=0.85,
            marker="o", label="Acquired inactive",
        )
        ax.scatter(
            X_pool[active, 0], X_pool[active, 1], s=22, c=COLORS[strategy],
            linewidths=0, marker="o", label="Acquired active",
        )
        ax.scatter(
            X_pool[initial, 0], X_pool[initial, 1], s=28, c="#202020",
            marker="x", linewidths=1.0, label="Initial set",
        )
        ax.set_title(DISPLAY_NAMES[strategy])
        ax.set_xlabel("Feature 1")
        ax.grid(alpha=0.15, linewidth=0.6)

    axes[0].set_ylabel("Feature 2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return seed


def summarise_quantiles(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        data.groupby(["strategy", "round"])[value_column]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def plot_roc_auc(
    history: pd.DataFrame,
    output_path: Path,
    initialization: str = "random",
) -> None:
    subset = history[
        (history["initialization_strategy"] == initialization)
        & (history["strategy"].isin(STRATEGIES))
    ].copy()
    summary = summarise_quantiles(subset, "roc_auc")
    fig, ax = plt.subplots(figsize=(6.7, 4.15), constrained_layout=True)

    for strategy in STRATEGIES:
        values = summary[summary["strategy"] == strategy].sort_values("round")
        ax.fill_between(
            values["round"], values["q25"], values["q75"],
            color=COLORS[strategy], alpha=0.11, linewidth=0,
        )
        ax.plot(
            values["round"], values["median"], color=COLORS[strategy],
            linestyle=LINESTYLES[strategy], linewidth=2.0, marker="o",
            markersize=3.5, label=DISPLAY_NAMES[strategy],
        )

    ax.set_xlabel("Acquisition round")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(sorted(summary["round"].unique()))
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    lower, upper = summary["q25"].min(), summary["q75"].max()
    padding = max(0.004, 0.08 * (upper - lower))
    ax.set_ylim(lower - padding, upper + padding)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def mean_within_batch_nn_distance(selected_indices: object, X_pool: np.ndarray) -> float:
    selected = parse_indices(selected_indices)
    if selected.size < 2:
        return np.nan
    batch = X_pool[selected]
    distances = cdist(batch, batch, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    return float(distances.min(axis=1).mean())


def plot_batch_redundancy(
    history: pd.DataFrame,
    X_pool: np.ndarray,
    output_path: Path,
    initialization: str = "random",
) -> pd.DataFrame:
    subset = history[
        (history["initialization_strategy"] == initialization)
        & (history["strategy"].isin(STRATEGIES))
        & (history["round"] > 0)
    ].copy()
    subset["mean_batch_nn_distance"] = subset["selected_indices"].apply(
        lambda value: mean_within_batch_nn_distance(value, X_pool)
    )
    summary = summarise_quantiles(subset, "mean_batch_nn_distance")
    fig, ax = plt.subplots(figsize=(6.7, 4.15), constrained_layout=True)

    for strategy in STRATEGIES:
        values = summary[summary["strategy"] == strategy].sort_values("round")
        ax.fill_between(
            values["round"], values["q25"], values["q75"],
            color=COLORS[strategy], alpha=0.11, linewidth=0,
        )
        ax.plot(
            values["round"], values["median"], color=COLORS[strategy],
            linestyle=LINESTYLES[strategy], linewidth=2.0, marker="o",
            markersize=3.5, label=DISPLAY_NAMES[strategy],
        )

    ax.set_xlabel("Acquisition round")
    ax.set_ylabel("Within-batch nearest-neighbour distance")
    ax.set_xticks(sorted(summary["round"].unique()))
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return subset[[
        "initialization_strategy", "strategy", "seed", "round",
        "mean_batch_nn_distance",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the minimal figure set for Part I."
    )
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--initialization", choices=["random", "diverse"], default="random"
    )
    args = parser.parse_args()

    configure_style()
    history = load_history(args.history)
    X_pool, y_pool = make_pool()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    representative_seed = plot_acquisition_behaviour(
        history, X_pool, y_pool,
        args.output_dir / "toy_article_acquisition_behaviour.svg",
        initialization=args.initialization,
        target_round=10,
    )
    plot_roc_auc(
        history,
        args.output_dir / "toy_article_roc_auc.svg",
        initialization=args.initialization,
    )
    diversity_table = plot_batch_redundancy(
        history, X_pool,
        args.output_dir / "toy_article_batch_redundancy.svg",
        initialization=args.initialization,
    )
    diversity_table.to_csv(
        table_dir / "toy_article_batch_redundancy.csv", index=False
    )

    print("Generated Part I figures:")
    print(
        " -", args.output_dir / "toy_article_acquisition_behaviour.svg",
        f"(representative seed: {representative_seed})"
    )
    print(" -", args.output_dir / "toy_article_roc_auc.svg")
    print(" -", args.output_dir / "toy_article_batch_redundancy.svg")
    print("Derived table:")
    print(" -", table_dir / "toy_article_batch_redundancy.csv")


if __name__ == "__main__":
    main()