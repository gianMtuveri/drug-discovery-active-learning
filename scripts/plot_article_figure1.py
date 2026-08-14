"""Generate Figure 1 for the active-learning article.

The figure compares Random, Greedy, and Uncertainty acquisition after the same
number of rounds, using the same visual palette in every panel:

- untested pool: light grey
- initial labelled set: blue
- newly acquired active samples: green
- newly acquired inactive samples: red
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.toy_data import make_toy_dataset


HISTORY_PATH = Path("results/tables/toy_initialization_comparison_history.csv")
OUTPUT_DIR = Path("results/figures/classification")

STRATEGIES = ["random", "greedy", "uncertainty_topk"]

DISPLAY_NAMES = {
    "random": "Random",
    "greedy": "Greedy",
    "uncertainty_topk": "Uncertainty",
}

PALETTE = {
    "untested": "#D9D9D9",
    "initial": "#0000CC",
    "active": "#008000",
    "inactive": "#FF0000",
    "edge": "#202020",
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
    """Load and validate the saved toy active-learning history."""
    if not path.exists():
        raise FileNotFoundError(
            f"History file not found: {path}\n"
            "Run scripts/run_toy_repeated_simulation.py first."
        )

    history = pd.read_csv(path)
    required = {
        "round",
        "strategy",
        "seed",
        "initialization_strategy",
        "roc_auc",
        "initial_indices",
        "labeled_indices",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            "History is missing columns: " + ", ".join(sorted(missing))
        )
    return history


def make_pool() -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the pool used by the toy active-learning experiments."""
    X, y = make_toy_dataset(n_samples=1000, random_state=42)
    X_pool, _, y_pool, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    return X_pool, y_pool


def configure_style() -> None:
    """Set publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
        }
    )


def choose_representative_seed(
    history: pd.DataFrame,
    initialization: str,
    strategies: Iterable[str],
    target_round: int,
) -> int:
    """Choose one seed with near-median performance across all strategies."""
    subset = history[
        (history["initialization_strategy"] == initialization)
        & (history["strategy"].isin(strategies))
        & (history["round"] == target_round)
    ].copy()

    if subset.empty:
        raise ValueError(
            "No campaigns were found for "
            f"initialization={initialization!r}, round={target_round}."
        )

    medians = subset.groupby("strategy")["roc_auc"].median()
    subset["absolute_deviation"] = subset.apply(
        lambda row: abs(row["roc_auc"] - medians.loc[row["strategy"]]),
        axis=1,
    )
    seed_scores = subset.groupby("seed")["absolute_deviation"].mean()
    if seed_scores.empty:
        raise ValueError("No complete final-round campaigns were found.")
    return int(seed_scores.idxmin())


def plot_figure1(
    history: pd.DataFrame,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    output_dir: Path,
    initialization: str = "random",
    target_round: int = 10,
    seed: int | None = None,
) -> int:
    """Create the three-panel acquisition-strategy comparison."""
    if seed is None:
        seed = choose_representative_seed(
            history=history,
            initialization=initialization,
            strategies=STRATEGIES,
            target_round=target_round,
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.6),
        sharex=True,
        sharey=True,
    )

    for ax, strategy, panel_letter in zip(axes, STRATEGIES, ["A", "B", "C"]):
        rows = history[
            (history["initialization_strategy"] == initialization)
            & (history["strategy"] == strategy)
            & (history["seed"] == seed)
            & (history["round"] == target_round)
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected one row for strategy={strategy}, seed={seed}, "
                f"round={target_round}; found {len(rows)}."
            )

        row = rows.iloc[0]
        initial = parse_indices(row["initial_indices"])
        labelled = parse_indices(row["labeled_indices"])
        acquired = np.setdiff1d(labelled, initial, assume_unique=False)
        untested = np.setdiff1d(
            np.arange(len(X_pool)), labelled, assume_unique=False
        )

        acquired_active = acquired[y_pool[acquired] == 1]
        acquired_inactive = acquired[y_pool[acquired] == 0]

        ax.scatter(
            X_pool[untested, 0],
            X_pool[untested, 1],
            s=15,
            color=PALETTE["untested"],
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            label="Untested",
            zorder=1,
        )
        ax.scatter(
            X_pool[initial, 0],
            X_pool[initial, 1],
            s=36,
            color=PALETTE["initial"],
            edgecolors=PALETTE["edge"],
            linewidths=0.55,
            marker="o",
            label="Initial set",
            zorder=4,
        )
        ax.scatter(
            X_pool[acquired_active, 0],
            X_pool[acquired_active, 1],
            s=39,
            color=PALETTE["active"],
            edgecolors=PALETTE["edge"],
            linewidths=0.55,
            marker="o",
            label="New active",
            zorder=3,
        )
        ax.scatter(
            X_pool[acquired_inactive, 0],
            X_pool[acquired_inactive, 1],
            s=39,
            color=PALETTE["inactive"],
            edgecolors=PALETTE["edge"],
            linewidths=0.55,
            marker="o",
            label="New inactive",
            zorder=3,
        )

        ax.set_title(DISPLAY_NAMES[strategy], pad=10)
        ax.set_xlabel("Feature 1")
        ax.grid(alpha=0.16, linewidth=0.6)
        ax.text(
            -0.09,
            1.04,
            panel_letter,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    axes[0].set_ylabel("Feature 2")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.5,
        columnspacing=1.6,
    )

    fig.suptitle(
        f"Campaign state after {target_round} active-learning rounds",
        fontsize=16,
        y=0.94,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.13,
        top=0.80,
        wspace=0.08,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "article_figure1"

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            bbox_inches="tight",
        )

    plt.close(fig)
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 1 comparing toy active-learning "
            "acquisition strategies."
        )
    )
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--initialization",
        choices=["random", "diverse"],
        default="random",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=10,
        dest="target_round",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    configure_style()
    history = load_history(args.history)
    X_pool, y_pool = make_pool()

    representative_seed = plot_figure1(
        history=history,
        X_pool=X_pool,
        y_pool=y_pool,
        output_dir=args.output_dir,
        initialization=args.initialization,
        target_round=args.target_round,
        seed=args.seed,
    )

    print("Generated Figure 1:")
    print(" -", args.output_dir / "article_figure1.png")
    print(" -", args.output_dir / "article_figure1.pdf")
    print(" -", args.output_dir / "article_figure1.svg")
    print("Representative seed:", representative_seed)


if __name__ == "__main__":
    main()
