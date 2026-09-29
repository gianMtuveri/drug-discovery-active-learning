#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Plot Figure 12: v0.5 multi-objective results.")
parser.add_argument(
    "--input",
    type=Path,
    default=Path("results/regression/analysis/v050_egfr/summary_complete_campaigns.csv"),
    help="Complete-campaign summary CSV.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("results/figures/regression"),
    help="Directory for publication figure outputs.",
)
args = parser.parse_args()

INPUT = args.input
OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = "figure12_multiobjective_results"


# ---------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------

MODEL_ORDER = [
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
]

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "bayesian_ridge": "Bayesian Ridge",
    "gaussian_process": "Gaussian Process",
}

# Keep paired strategies adjacent:
# baseline → +QED, diversity → diversity+QED
STRATEGY_ORDER = [
    "ucb",
    "ucb_qed",
    "ucb_diversity",
    "ucb_qed_diversity",
]

STRATEGY_LABELS = {
    "ucb": "UCB",
    "ucb_qed": "UCB + QED",
    "ucb_diversity": "UCB + Diversity",
    "ucb_qed_diversity": "UCB + QED + Diversity",
}


# ---------------------------------------------------------------------
# Factorial visual encoding
#
# Colour  = QED
# Hatching = diversity
# ---------------------------------------------------------------------

NO_QED_COLOR = "#4C78A8"
QED_COLOR = "#F58518"

COLORS = {
    "ucb": NO_QED_COLOR,
    "ucb_qed": QED_COLOR,
    "ucb_diversity": NO_QED_COLOR,
    "ucb_qed_diversity": QED_COLOR,
}

HATCHES = {
    "ucb": "",
    "ucb_qed": "",
    "ucb_diversity": "///",
    "ucb_qed_diversity": "///",
}


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "hatch.linewidth": 0.8,
    }
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT)

expected = {
    "strategy",
    "model",
    "joint_discovery_count_mean",
    "joint_discovery_count_sd",
    "affinity_discovery_count_mean",
    "affinity_discovery_count_sd",
    "joint_scaffold_count_mean",
    "joint_scaffold_count_sd",
    "rmse_mean",
    "rmse_sd",
    "pearson_mean",
    "pearson_sd",
    "qed_fraction_of_affinity_discoveries_mean",
    "qed_fraction_of_affinity_discoveries_sd",
}

missing = expected - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns in {INPUT}: {sorted(missing)}"
    )


# ---------------------------------------------------------------------
# Generic grouped-bar function
# ---------------------------------------------------------------------

def grouped_bars(
    ax,
    mean_col,
    sd_col,
    ylabel,
    title,
    zero_baseline=True,
):
    x = np.arange(len(MODEL_ORDER))
    width = 0.19

    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for offset, strategy in zip(offsets, STRATEGY_ORDER):
        means = []
        sds = []

        for model in MODEL_ORDER:
            row = df[
                (df["model"] == model)
                & (df["strategy"] == strategy)
            ]

            if len(row) != 1:
                raise ValueError(
                    f"Expected exactly one row for {model}/{strategy}; "
                    f"found {len(row)}"
                )

            means.append(float(row.iloc[0][mean_col]))
            sds.append(float(row.iloc[0][sd_col]))

        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=sds,
            capsize=2,
            color=COLORS[strategy],
            hatch=HATCHES[strategy],
            edgecolor="#333333",
            linewidth=0.45,
            error_kw={
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
        )

    ax.set_xticks(
        x,
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")

    if zero_baseline:
        ax.set_ylim(bottom=0)

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(13.0, 7.2),
    constrained_layout=False,
)

ax_a, ax_b, ax_c = axes[0]
ax_d, ax_e, ax_f = axes[1]


# A — final joint discovery
grouped_bars(
    ax_a,
    "joint_discovery_count_mean",
    "joint_discovery_count_sd",
    "Joint discoveries",
    "A   Joint discovery",
)


# B — affinity-only discovery
grouped_bars(
    ax_b,
    "affinity_discovery_count_mean",
    "affinity_discovery_count_sd",
    "Affinity-qualified discoveries",
    "B   Affinity discovery",
)


# C — scaffold coverage
grouped_bars(
    ax_c,
    "joint_scaffold_count_mean",
    "joint_scaffold_count_sd",
    "Distinct qualifying scaffolds",
    "C   Structural coverage",
)


# D — fraction of affinity discoveries also satisfying QED
grouped_bars(
    ax_d,
    "qed_fraction_of_affinity_discoveries_mean",
    "qed_fraction_of_affinity_discoveries_sd",
    "Fraction of affinity discoveries",
    "D   QED-qualified fraction",
)

ax_d.set_ylim(0, 1)


# E — RMSE
grouped_bars(
    ax_e,
    "rmse_mean",
    "rmse_sd",
    "Held-out RMSE",
    "E   Predictive error",
    zero_baseline=False,
)


# F — Pearson
grouped_bars(
    ax_f,
    "pearson_mean",
    "pearson_sd",
    "Held-out Pearson r",
    "F   Predictive correlation",
    zero_baseline=False,
)


# ---------------------------------------------------------------------
# Shared legend
#
# Keep the four combinations explicit while allowing the visual grammar
# to reveal the factorial design:
#
# colour  -> QED
# hatch   -> diversity
# ---------------------------------------------------------------------

handles = [
    Patch(
        facecolor=COLORS[strategy],
        edgecolor="#333333",
        linewidth=0.45,
        hatch=HATCHES[strategy],
        label=STRATEGY_LABELS[strategy],
    )
    for strategy in STRATEGY_ORDER
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=4,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.8,
)


# ---------------------------------------------------------------------
# Figure-level definition
# ---------------------------------------------------------------------

fig.text(
    0.5,
    0.995,
    "Joint discovery: affinity ≤ 500 nM and QED ≥ 0.6",
    ha="center",
    va="top",
    fontsize=10,
)


fig.subplots_adjust(
    left=0.07,
    right=0.985,
    bottom=0.10,
    top=0.88,
    wspace=0.32,
    hspace=0.42,
)


# ---------------------------------------------------------------------
# Save all formats
# ---------------------------------------------------------------------

for suffix in ("png", "svg", "pdf"):
    path = OUTPUT_DIR / f"{OUTPUT_NAME}.{suffix}"

    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=300 if suffix == "png" else None,
    )

    print(f"Saved: {path}")

plt.close(fig)
