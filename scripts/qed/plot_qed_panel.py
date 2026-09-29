#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import Draw


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TARGET_DIR = Path("data/processed/targets/EGFR")

CLEAN_PATH = TARGET_DIR / "clean.parquet"
DESCRIPTORS_PATH = TARGET_DIR / "descriptors.parquet"

parser = argparse.ArgumentParser(description="Plot Figure 11: QED multi-objective concept.")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("results/figures/regression"),
    help="Directory for publication figure outputs.",
)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = "figure11_qed_multiobjective"


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

AFFINITY_THRESHOLD = 500.0
QED_THRESHOLD = 0.6

# Acquisition-strategy colours, shared with the results figure:
# blue = no QED; orange = +QED.
BLUE = "#4C78A8"
ORANGE = "#F58518"

# Panel-B colours describe molecular outcome classes and are deliberately
# independent from the acquisition-strategy colour grammar.
NEITHER_COLOR = "#D0D0D0"
AFFINITY_ONLY_COLOR = "#6F6F6F"
QED_ONLY_COLOR = "#9C814D"
JOINT_COLOR = "#4E9A51"

TEXT = "#222222"
LIGHT_BLUE = "#EAF2F8"
LIGHT_ORANGE = "#FFF1E3"


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


# ---------------------------------------------------------------------
# Load EGFR data
# ---------------------------------------------------------------------

clean = pd.read_parquet(CLEAN_PATH)
desc = pd.read_parquet(DESCRIPTORS_PATH)

if len(clean) != len(desc):
    raise ValueError(
        f"Alignment failure: clean={len(clean)}, descriptors={len(desc)}"
    )

if "QED" not in desc.columns:
    raise ValueError(
        f"'QED' not found in {DESCRIPTORS_PATH}. "
        f"Available columns: {list(desc.columns)}"
    )

affinity = clean["aff_nM_median"].to_numpy(dtype=float)
qed = desc["QED"].to_numpy(dtype=float)

valid = (
    np.isfinite(affinity)
    & (affinity > 0)
    & np.isfinite(qed)
)

affinity = affinity[valid]
qed = qed[valid]

print(f"Valid EGFR molecules: {len(affinity)}")


# ---------------------------------------------------------------------
# Representative molecule for panel A
# ---------------------------------------------------------------------

# Use a real EGFR-pool molecule closest to the median QED.
valid_clean = clean.loc[valid].reset_index(drop=True)
valid_desc = desc.loc[valid].reset_index(drop=True)

median_qed = valid_desc["QED"].median()

representative_idx = (
    valid_desc["QED"] - median_qed
).abs().idxmin()

representative_smiles = valid_clean.loc[
    representative_idx, "canonical_smiles"
]

representative_qed = float(
    valid_desc.loc[representative_idx, "QED"]
)

representative_affinity = float(
    valid_clean.loc[representative_idx, "aff_nM_median"]
)

mol = Chem.MolFromSmiles(representative_smiles)

if mol is None:
    raise ValueError("Could not parse representative EGFR molecule.")

mol_img = Draw.MolToImage(
    mol,
    size=(600, 400),
)

print("\nRepresentative molecule")
print("-----------------------")
print(f"Pool index: {representative_idx}")
print(f"SMILES: {representative_smiles}")
print(f"QED: {representative_qed:.3f}")
print(f"Affinity: {representative_affinity:.3g} nM")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def add_box(
    ax,
    xy,
    width,
    height,
    text,
    facecolor="white",
    edgecolor="#CCCCCC",
    fontsize=9,
    weight="normal",
):
    x, y = xy

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        transform=ax.transAxes,
    )

    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.5,
        color="#777777",
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(14, 8))

gs = GridSpec(
    2,
    2,
    figure=fig,
    width_ratios=[1.05, 1.45],
    height_ratios=[1.45, 0.85],
    hspace=0.22,
    wspace=0.18,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])


# =====================================================================
# A — From molecular structure to QED
# =====================================================================

ax_a.set_axis_off()

ax_a.set_title(
    "A   From molecular structure to QED",
    loc="left",
    fontweight="bold",
    pad=12,
)

# ------------------------------------------------------------------
# Molecular structure
# ------------------------------------------------------------------

mol_ax = ax_a.inset_axes([0.00, 0.34, 0.27, 0.36])
mol_ax.imshow(mol_img)
mol_ax.axis("off")

# Replace this only after the source BindingDB record has been used to
# resolve a genuine compound name. Do not infer a name from the SMILES.
REPRESENTATIVE_NAME = "EGFR pool molecule"

ax_a.text(
    0.135,
    0.285,
    REPRESENTATIVE_NAME,
    transform=ax_a.transAxes,
    ha="center",
    va="top",
    fontweight="bold",
    fontsize=9,
)


# ------------------------------------------------------------------
# Structure -> descriptors
# ------------------------------------------------------------------

add_arrow(
    ax_a,
    (0.27, 0.52),
    (0.315, 0.52),
)

descriptor_box = FancyBboxPatch(
    (0.325, 0.255),
    0.47,
    0.53,
    boxstyle="round,pad=0.018,rounding_size=0.025",
    facecolor="#F7F9FA",
    edgecolor="#D5DADD",
    linewidth=0.9,
    transform=ax_a.transAxes,
)

ax_a.add_patch(descriptor_box)

ax_a.text(
    0.56,
    0.705,
    "Molecular properties",
    transform=ax_a.transAxes,
    ha="center",
    va="center",
    fontweight="bold",
    fontsize=9,
)

properties = [
    "Molecular weight",
    "Lipophilicity",
    "Polar surface area",
    "H-bond donors",
    "H-bond acceptors",
    "Rotatable bonds",
    "Aromaticity",
    "Structural alerts",
]

# A single column is easier to scan and avoids crowding between labels.
property_y = [0.625, 0.580, 0.535, 0.490, 0.445, 0.400, 0.355, 0.310]

for y, label in zip(property_y, properties):
    ax_a.text(
        0.56,
        y,
        label,
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
    )


# ------------------------------------------------------------------
# Descriptors -> QED
# ------------------------------------------------------------------

add_arrow(
    ax_a,
    (0.80, 0.52),
    (0.835, 0.52),
)

qed_box = FancyBboxPatch(
    (0.84, 0.395),
    0.13,
    0.24,
    boxstyle="round,pad=0.018,rounding_size=0.025",
    facecolor=LIGHT_ORANGE,
    edgecolor=ORANGE,
    linewidth=1.2,
    transform=ax_a.transAxes,
)

ax_a.add_patch(qed_box)

ax_a.text(
    0.905,
    0.555,
    "QED",
    transform=ax_a.transAxes,
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
)

ax_a.text(
    0.905,
    0.47,
    "0 – 1",
    transform=ax_a.transAxes,
    ha="center",
    va="center",
    fontsize=10,
)


# ------------------------------------------------------------------
# Main conceptual message
# ------------------------------------------------------------------

ax_a.text(
    0.54,
    0.145,
    "Structure-derived molecular quality\n"
    "available before experimental acquisition",
    transform=ax_a.transAxes,
    ha="center",
    va="center",
    fontsize=9,
    color="#444444",
)

# =====================================================================
# B — Real EGFR affinity × QED landscape
# =====================================================================

ax_b.set_title(
    "B   Affinity and QED define different objectives",
    loc="left",
    fontweight="bold",
    pad=12,
)


# ------------------------------------------------------------------
# Classify the actual EGFR pool
# ------------------------------------------------------------------

strong = affinity <= AFFINITY_THRESHOLD
high_qed = qed >= QED_THRESHOLD

groups = {
    "Neither": (~strong) & (~high_qed),
    "Affinity only": strong & (~high_qed),
    "QED only": (~strong) & high_qed,
    "Joint discovery": strong & high_qed,
}

group_styles = {
    "Neither": {
        "color": NEITHER_COLOR,
        "alpha": 0.18,
        "s": 7,
    },
    "Affinity only": {
        "color": AFFINITY_ONLY_COLOR,
        "alpha": 0.30,
        "s": 8,
    },
    "QED only": {
        "color": QED_ONLY_COLOR,
        "alpha": 0.34,
        "s": 8,
    },
    "Joint discovery": {
        "color": JOINT_COLOR,
        "alpha": 0.52,
        "s": 9,
    },
}


# ------------------------------------------------------------------
# Joint-discovery region
# ------------------------------------------------------------------

# Explicit log-decade limits ensure the shaded joint-discovery region
# fills the complete visible upper-left quadrant.
x_plot_min = 10 ** np.floor(np.log10(np.nanmin(affinity)))
x_plot_max = 10 ** np.ceil(np.log10(np.nanmax(affinity)))

ax_b.set_xscale("log")
ax_b.set_xlim(x_plot_min, x_plot_max)
ax_b.set_ylim(0, 1)

ax_b.axvspan(
    x_plot_min,
    AFFINITY_THRESHOLD,
    ymin=QED_THRESHOLD,
    ymax=1.0,
    color=JOINT_COLOR,
    alpha=0.08,
    linewidth=0,
    zorder=0,
)


# ------------------------------------------------------------------
# Plot groups separately
# ------------------------------------------------------------------

for label in [
    "Neither",
    "Affinity only",
    "QED only",
    "Joint discovery",
]:
    mask = groups[label]
    style = group_styles[label]

    ax_b.scatter(
        affinity[mask],
        qed[mask],
        s=style["s"],
        alpha=style["alpha"],
        color=style["color"],
        linewidths=0,
        rasterized=True,
        label=f"{label} (n={mask.sum():,})",
    )


# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------

ax_b.axvline(
    AFFINITY_THRESHOLD,
    color="#333333",
    linestyle="--",
    linewidth=1.0,
)

ax_b.axhline(
    QED_THRESHOLD,
    color="#333333",
    linestyle="--",
    linewidth=1.0,
)

ax_b.set_xlabel("Experimental affinity (nM)")
ax_b.set_ylabel("QED")

ax_b.grid(
    alpha=0.10,
    linewidth=0.5,
)

ax_b.set_axisbelow(True)


# ------------------------------------------------------------------
# Threshold labels
# ------------------------------------------------------------------

ax_b.text(
    AFFINITY_THRESHOLD,
    1.015,
    "500 nM",
    ha="center",
    va="bottom",
    fontsize=8,
)

ax_b.text(
    ax_b.get_xlim()[1],
    QED_THRESHOLD + 0.015,
    "QED = 0.6",
    ha="right",
    va="bottom",
    fontsize=8,
)


# ------------------------------------------------------------------
# Region annotation
# ------------------------------------------------------------------

ax_b.text(
    0.035,
    0.94,
    "Joint discovery",
    transform=ax_b.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    fontweight="bold",
    color=JOINT_COLOR,
)


# ------------------------------------------------------------------
# Legend
# ------------------------------------------------------------------

ax_b.legend(
    loc="lower left",
    frameon=False,
    fontsize=7.4,
    markerscale=1.4,
    borderaxespad=0.2,
)


# =====================================================================
# C — From single- to multi-objective acquisition
# =====================================================================

ax_c.set_axis_off()

ax_c.set_title(
    "C   From single- to multi-objective acquisition",
    loc="left",
    fontweight="bold",
    pad=10,
)

# Panel C is a factorial 2 × 2 design, not a sequential workflow.
# Rows encode QED; columns encode diversity.

left = 0.16
right = 0.59
bottom = 0.12
top = 0.51
box_w = 0.30
box_h = 0.25

# Column headings
ax_c.text(
    left + box_w / 2,
    0.815,
    "No diversity",
    transform=ax_c.transAxes,
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#555555",
)
ax_c.text(
    right + box_w / 2,
    0.815,
    "+ Diversity",
    transform=ax_c.transAxes,
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#555555",
)

# Row headings
ax_c.text(
    0.125,
    top + box_h / 2,
    "No QED",
    transform=ax_c.transAxes,
    ha="right",
    va="center",
    fontsize=9,
    fontweight="bold",
    color=BLUE,
)
ax_c.text(
    0.125,
    bottom + box_h / 2,
    "+ QED",
    transform=ax_c.transAxes,
    ha="right",
    va="center",
    fontsize=9,
    fontweight="bold",
    color=ORANGE,
)

cells = [
    # x, y, title, body, face, edge, diversity
    (
        left, top,
        "UCB",
        "Predicted affinity\n+ uncertainty",
        LIGHT_BLUE, BLUE, False,
    ),
    (
        right, top,
        "UCB + Diversity",
        "Predicted affinity + uncertainty\n+ structural diversity",
        LIGHT_BLUE, BLUE, True,
    ),
    (
        left, bottom,
        "UCB + QED",
        "Predicted affinity + uncertainty\n+ molecular quality",
        LIGHT_ORANGE, ORANGE, False,
    ),
    (
        right, bottom,
        "UCB + QED + Diversity",
        "Predicted affinity + uncertainty\n+ molecular quality + structural diversity",
        LIGHT_ORANGE, ORANGE, True,
    ),
]

for x, y, title, body, face, edge, has_diversity in cells:
    patch = FancyBboxPatch(
        (x, y),
        box_w,
        box_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.15,
        alpha=0.68,
        transform=ax_c.transAxes,
    )
    ax_c.add_patch(patch)

    ax_c.text(
        x + box_w / 2,
        y + box_h - 0.065,
        title,
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color=edge,
    )

    ax_c.text(
        x + box_w / 2,
        y + 0.095,
        body,
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=1.3,
        color=TEXT,
    )




# Minimal guides emphasise the factorial axes rather than implying sequence.
ax_c.annotate(
    "",
    xy=(0.925, 0.905),
    xytext=(0.15, 0.905),
    xycoords=ax_c.transAxes,
    arrowprops=dict(
        arrowstyle="-|>",
        color="#999999",
        linewidth=0.9,
    ),
)
ax_c.text(
    0.535,
    0.955,
    "Diversity objective",
    transform=ax_c.transAxes,
    ha="center",
    va="center",
    fontsize=7.5,
    color="#777777",
)

ax_c.annotate(
    "",
    xy=(0.055, 0.10),
    xytext=(0.055, 0.775),
    xycoords=ax_c.transAxes,
    arrowprops=dict(
        arrowstyle="-|>",
        color="#999999",
        linewidth=0.9,
    ),
)
ax_c.text(
    0.018,
    0.44,
    "QED objective",
    transform=ax_c.transAxes,
    ha="center",
    va="center",
    rotation=90,
    fontsize=7.5,
    color="#777777",
)


# ---------------------------------------------------------------------
# Save
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
