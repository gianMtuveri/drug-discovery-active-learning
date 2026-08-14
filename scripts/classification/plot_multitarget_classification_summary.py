"""
Build Figure 6:
Multi-target evaluation of active-learning acquisition strategies
for the classification model.

Expected inputs
---------------
For every target:

    data/processed/targets/<TARGET>/metadata.json
    data/processed/targets/<TARGET>/descriptors.parquet
    results/tables/<target>_classification_summary.csv

The script uses the final acquisition round and, by default, the
diversity-based initialization for every strategy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


DEFAULT_TARGETS = [
    "EGFR", "JAK2", "PARP1", "BRAF", "ABL1",
    "SRC", "VEGFR2", "CDK2", "DRD2", "CA2",
]

STRATEGY_ORDER = [
    "greedy", "random", "uncertainty_topk", "uncertainty_diverse"
]

STRATEGY_LABELS = {
    "greedy": "Greedy",
    "random": "Random",
    "uncertainty_topk": "Top-K",
    "uncertainty_diverse": "Diverse",
}

STRATEGY_ALIASES = {
    "greedy": "greedy",
    "random": "random",
    "topk": "uncertainty_topk",
    "top_k": "uncertainty_topk",
    "uncertainty_topk": "uncertainty_topk",
    "uncertainty_top_k": "uncertainty_topk",
    "uncertainty": "uncertainty_topk",
    "diverse": "uncertainty_diverse",
    "uncertainty_diverse": "uncertainty_diverse",
    "uncertainty_diversity": "uncertainty_diverse",
}

DESCRIPTOR_SPECS = [
    ("MolWt", "Molecular weight\n(Da)", ["MolWt", "mol_wt", "molecular_weight", "mw"]),
    ("TPSA", "TPSA\n(Å²)", ["TPSA", "tpsa", "topological_polar_surface_area"]),
]

CORRELATION_SPECS = [
    ("MolWt", ["MolWt", "mol_wt", "molecular_weight", "mw"]),
    ("LogP", ["LogP", "MolLogP", "logp", "xlogp"]),
    ("TPSA", ["TPSA", "tpsa", "topological_polar_surface_area"]),
    (
        "RotatableBonds",
        ["RotatableBonds", "NumRotatableBonds", "rotatable_bonds", "n_rotatable_bonds"],
    ),
    ("RingCount", ["RingCount", "ring_count", "NumRings", "n_rings"]),
    (
        "FractionCSP3",
        ["FractionCSP3", "fraction_csp3", "fraction_sp3", "frac_csp3", "fsp3"],
    ),
]

COLUMN_ALIASES = {
    "strategy": ["strategy", "acquisition_strategy", "acquisition", "query_strategy"],
    "initialization": [
        "initialization_strategy", "initialization", "init_strategy",
        "initialisation_strategy", "initialisation",
    ],
    "round": ["round", "acquisition_round", "iteration"],
    "roc_auc_mean": ["roc_auc_mean", "auc_mean", "mean_roc_auc", "rocauc_mean"],
    "roc_auc_std": ["roc_auc_std", "auc_std", "std_roc_auc", "rocauc_std"],
    "actives_mean": [
        "actives_discovered_mean", "actives_found_mean", "cumulative_actives_mean",
        "mean_actives_discovered", "mean_actives_found", "actives_mean",
    ],
    "actives_std": [
        "actives_discovered_std", "actives_found_std", "cumulative_actives_std",
        "std_actives_discovered", "std_actives_found", "actives_std",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--target-root", type=Path, default=Path("data/processed/targets")
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/tables")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--initialization", default="diverse")
    parser.add_argument(
        "--round", type=int, default=None,
        help="Acquisition round to plot. Default: final available round per target.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def canonicalize_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"[^a-z0-9_]+", "", text)


def resolve_column(
    columns: Iterable[str], aliases: Iterable[str], required: bool = True
) -> str | None:
    lookup = {canonicalize_name(column): column for column in columns}
    for alias in aliases:
        key = canonicalize_name(alias)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Could not resolve a required column. Tried {list(aliases)}. "
            f"Available: {list(columns)}"
        )
    return None


def normalize_strategy(value: object) -> str:
    key = canonicalize_name(value)
    return STRATEGY_ALIASES.get(key, key)


def normalize_initialization(value: object) -> str:
    key = canonicalize_name(value)
    if key in {"diverse", "diversity", "diversity_based", "diverse_initialization"}:
        return "diverse"
    if key in {"random", "random_initialization"}:
        return "random"
    return key


def find_summary_file(results_dir: Path, target: str) -> Path:
    candidates = [
        results_dir / f"{target.lower()}_classification_summary.csv",
        results_dir / f"{target}_classification_summary.csv",
        results_dir / target / "classification_summary.csv",
        results_dir / target.lower() / "classification_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(results_dir.glob(f"**/*{target.lower()}*classification*summary*.csv"))
    if not matches:
        matches = sorted(results_dir.glob(f"**/*{target}*classification*summary*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No classification summary CSV found for {target} in {results_dir}."
    )


def find_target_dir(target_root: Path, target: str) -> Path:
    for candidate in (target_root / target, target_root / target.lower()):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No processed target directory found for {target} in {target_root}."
    )


def get_nested(metadata: dict, paths: list[tuple[str, ...]], default=np.nan):
    """Return the first value found at one of the requested metadata paths."""
    for path in paths:
        current = metadata
        valid = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                valid = False
                break
            current = current[key]
        if valid:
            return current
    return default


def find_key_recursive(mapping: object, key: str, default=np.nan):
    """Find a key anywhere in a nested metadata dictionary."""
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        for value in mapping.values():
            result = find_key_recursive(value, key, default=default)
            if not (isinstance(result, float) and np.isnan(result)):
                return result
    elif isinstance(mapping, list):
        for value in mapping:
            result = find_key_recursive(value, key, default=default)
            if not (isinstance(result, float) and np.isnan(result)):
                return result
    return default


def load_metadata(target_dir: Path, target: str) -> dict:
    path = target_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    molecules = get_nested(
        metadata,
        [
            ("n_molecules_raw",),
            ("n_molecules",),
            ("n_samples",),
            ("num_molecules",),
            ("molecules",),
            ("dataset", "n_molecules_raw"),
            ("dataset", "n_molecules"),
            ("dataset", "n_samples"),
        ],
    )
    if pd.isna(pd.to_numeric(molecules, errors="coerce")):
        molecules = find_key_recursive(metadata, "n_molecules_raw")
    active_fraction = get_nested(
        metadata,
        [
            ("active_fraction",), ("class_balance", "active_fraction"),
            ("dataset", "active_fraction"),
        ],
    )
    threshold = get_nested(
        metadata,
        [
            ("threshold_nM",), ("threshold_nm",), ("activity_threshold_nM",),
            ("activity_threshold_nm",), ("classification_threshold_nM",),
            ("classification_threshold_nm",), ("threshold",),
            ("classification", "threshold_nM"), ("classification", "threshold"),
        ],
    )

    return {
        "Target": target,
        "Molecules": pd.to_numeric(molecules, errors="coerce"),
        "Active fraction": pd.to_numeric(active_fraction, errors="coerce"),
        "Threshold nM": pd.to_numeric(threshold, errors="coerce"),
    }


def load_descriptors(target_dir: Path, target: str) -> pd.DataFrame:
    path = target_dir / "descriptors.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing descriptor file: {path}")
    frame = pd.read_parquet(path).copy()
    frame["Target"] = target
    return frame


def standardize_summary(
    path: Path,
    target: str,
    initialization: str,
    selected_round: int | None,
) -> pd.DataFrame:
    raw = pd.read_csv(path)

    strategy_col = resolve_column(raw.columns, COLUMN_ALIASES["strategy"])
    init_col = resolve_column(raw.columns, COLUMN_ALIASES["initialization"], False)
    round_col = resolve_column(raw.columns, COLUMN_ALIASES["round"], False)
    roc_mean_col = resolve_column(raw.columns, COLUMN_ALIASES["roc_auc_mean"])
    roc_std_col = resolve_column(raw.columns, COLUMN_ALIASES["roc_auc_std"], False)
    actives_mean_col = resolve_column(raw.columns, COLUMN_ALIASES["actives_mean"])
    actives_std_col = resolve_column(raw.columns, COLUMN_ALIASES["actives_std"], False)

    frame = pd.DataFrame(
        {
            "target": target,
            "strategy": raw[strategy_col].map(normalize_strategy),
            "roc_auc_mean": pd.to_numeric(raw[roc_mean_col], errors="coerce"),
            "roc_auc_std": (
                pd.to_numeric(raw[roc_std_col], errors="coerce")
                if roc_std_col else np.nan
            ),
            "actives_mean": pd.to_numeric(raw[actives_mean_col], errors="coerce"),
            "actives_std": (
                pd.to_numeric(raw[actives_std_col], errors="coerce")
                if actives_std_col else np.nan
            ),
        }
    )

    if init_col:
        frame["initialization"] = raw[init_col].map(normalize_initialization)
        frame = frame.loc[
            frame["initialization"] == normalize_initialization(initialization)
        ].copy()
    else:
        frame["initialization"] = normalize_initialization(initialization)

    if round_col:
        frame["round"] = pd.to_numeric(raw[round_col], errors="coerce")
        round_to_use = selected_round
        if round_to_use is None:
            round_to_use = int(frame["round"].dropna().max())
        frame = frame.loc[frame["round"] == round_to_use].copy()
    else:
        frame["round"] = selected_round if selected_round is not None else np.nan

    frame = frame.loc[frame["strategy"].isin(STRATEGY_ORDER)].copy()

    if frame.groupby("strategy").size().max() > 1:
        numeric_columns = [
            "roc_auc_mean", "roc_auc_std", "actives_mean", "actives_std", "round"
        ]
        frame = (
            frame.groupby(["target", "strategy", "initialization"], as_index=False)[
                numeric_columns
            ]
            .mean(numeric_only=True)
        )

    missing = sorted(set(STRATEGY_ORDER) - set(frame["strategy"]))
    if missing:
        raise ValueError(
            f"{path} does not contain all expected strategies after filtering. "
            f"Missing: {missing}"
        )

    return frame


def build_descriptor_frame(
    descriptor_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str]]:
    combined = pd.concat(descriptor_frames, ignore_index=True)
    mapping: dict[str, str] = {}

    for canonical, _, aliases in DESCRIPTOR_SPECS:
        mapping[canonical] = resolve_column(combined.columns, aliases)
    for canonical, aliases in CORRELATION_SPECS:
        if canonical not in mapping:
            mapping[canonical] = resolve_column(combined.columns, aliases)

    return combined, mapping


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_dataset_table(ax: mpl.axes.Axes, metadata: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title("Benchmark datasets", pad=8)

    display = metadata.copy()
    display["Molecules"] = display["Molecules"].map(
        lambda value: f"{int(value):,}" if pd.notna(value) else "—"
    )
    display["Active fraction"] = display["Active fraction"].map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "—"
    )
    display["Threshold nM"] = display["Threshold nM"].map(
        lambda value: f"{value:g}" if pd.notna(value) else "—"
    )

    table = ax.table(
        cellText=display.values,
        colLabels=["Target", "Molecules", "Active frac.", "Threshold (nM)"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.00, 0.00, 1.00, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.18)

    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_text_props(weight="semibold")
            cell.set_facecolor("#eeeeee")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f8f8")


def plot_violin(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    target_order: list[str],
    column: str,
    title: str,
) -> None:
    values = [
        frame.loc[frame["Target"] == target, column].dropna().to_numpy()
        for target in target_order
    ]
    positions = np.arange(1, len(target_order) + 1)

    parts = ax.violinplot(
        values,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )

    for body in parts["bodies"]:
        body.set_facecolor("#4c9ad4")
        body.set_edgecolor("#4c9ad4")
        body.set_alpha(0.78)
    parts["cmedians"].set_color("#303030")
    parts["cmedians"].set_linewidth(1.0)

    ax.set_title(title, pad=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(target_order, rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.margins(x=0.02)


def annotated_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    title: str,
    value_format: str,
    colorbar_label: str,
) -> None:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="viridis")

    ax.set_title(title, pad=7)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [STRATEGY_LABELS.get(column, column) for column in matrix.columns],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    finite_values = values[np.isfinite(values)]
    midpoint = (
        float(np.nanmin(finite_values) + np.nanmax(finite_values)) / 2
        if finite_values.size else 0.5
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            ax.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="semibold",
                color="white" if value < midpoint else "black",
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.ax.set_ylabel(colorbar_label, rotation=90, va="bottom")


def calculate_ranks(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = results.copy()
    ranked["auc_rank"] = ranked.groupby("target")["roc_auc_mean"].rank(
        ascending=False, method="average"
    )
    ranked["actives_rank"] = ranked.groupby("target")["actives_mean"].rank(
        ascending=False, method="average"
    )
    ranked["combined_rank"] = ranked[["auc_rank", "actives_rank"]].mean(axis=1)

    summary = (
        ranked.groupby("strategy", as_index=False)
        .agg(
            mean_rank=("combined_rank", "mean"),
            rank_std=("combined_rank", "std"),
            auc_rank=("auc_rank", "mean"),
            actives_rank=("actives_rank", "mean"),
        )
        .sort_values("mean_rank")
    )
    return ranked, summary


def plot_average_rank(ax: mpl.axes.Axes, rank_summary: pd.DataFrame) -> None:
    display = rank_summary.sort_values("mean_rank", ascending=True)
    labels = [STRATEGY_LABELS.get(value, value) for value in display["strategy"]]
    y = np.arange(len(display))

    ax.barh(
        y,
        display["mean_rank"],
        xerr=display["rank_std"].fillna(0.0),
        color="#756bb1",
        alpha=0.88,
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank across targets and metrics\n(lower is better)")
    ax.set_title("Overall acquisition-strategy ranking", pad=7)
    ax.grid(axis="x", alpha=0.2)



def strategy_colors() -> dict[str, str]:
    return {
        "greedy": "#1f77b4",
        "random": "#ff7f0e",
        "uncertainty_topk": "#2ca02c",
        "uncertainty_diverse": "#d62728",
    }


def plot_auc_vs_actives(ax: mpl.axes.Axes, results: pd.DataFrame) -> None:
    colors = strategy_colors()

    for strategy in STRATEGY_ORDER:
        subset = results.loc[results["strategy"] == strategy]
        ax.scatter(
            subset["roc_auc_mean"],
            subset["actives_mean"],
            s=52,
            alpha=0.88,
            color=colors[strategy],
            edgecolor="white",
            linewidth=0.6,
            label=STRATEGY_LABELS[strategy],
            zorder=3,
        )

    # Connect the strategies evaluated on the same target.
    for _, subset in results.groupby("target"):
        subset = subset.sort_values("roc_auc_mean")
        ax.plot(
            subset["roc_auc_mean"],
            subset["actives_mean"],
            color="#bdbdbd",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )

    # Label each target once, at the centroid of its strategy-specific results.
    for target, subset in results.groupby("target"):
        x = float(subset["roc_auc_mean"].mean())
        y = float(subset["actives_mean"].mean())
        ax.annotate(
            target,
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#303030",
            ha="left",
            va="bottom",
            zorder=4,
        )

    ax.set_xlabel("Mean ROC-AUC")
    ax.set_ylabel("Mean active compounds discovered")
    ax.set_title("Predictive performance and hit discovery", pad=14)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="best")


def plot_correlation(
    ax: mpl.axes.Axes,
    descriptor_frame: pd.DataFrame,
    mapping: dict[str, str],
) -> None:
    labels = [canonical for canonical, _ in CORRELATION_SPECS]
    columns = [mapping[label] for label in labels]

    correlation = descriptor_frame[columns].corr(method="pearson")
    correlation.index = labels
    correlation.columns = labels

    values = correlation.to_numpy()
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax.set_title("Physicochemical descriptor correlations", pad=7)

    display_labels = ["MolWt", "LogP", "TPSA", "RotB", "Rings", "Fsp³"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if abs(value) > 0.55 else "black",
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.ax.set_ylabel("Pearson r", rotation=90, va="bottom")


def find_main_observations(
    results: pd.DataFrame,
    ranked: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> dict[str, str]:
    strategy_means = (
        results.groupby("strategy", as_index=False)
        .agg(roc_auc=("roc_auc_mean", "mean"), actives=("actives_mean", "mean"))
    )
    best_auc = strategy_means.loc[strategy_means["roc_auc"].idxmax(), "strategy"]
    best_actives = strategy_means.loc[strategy_means["actives"].idxmax(), "strategy"]

    consistency = (
        ranked.groupby("strategy")["combined_rank"]
        .std()
        .sort_values()
        .index[0]
    )
    best_overall = rank_summary.iloc[0]["strategy"]

    return {
        "Best predictive performance": STRATEGY_LABELS[best_auc],
        "Best active discovery": STRATEGY_LABELS[best_actives],
        "Best overall mean rank": STRATEGY_LABELS[best_overall],
        "Most consistent rank": STRATEGY_LABELS[consistency],
    }


def plot_main_findings(
    ax: mpl.axes.Axes,
    findings: dict[str, str],
    initialization: str,
    selected_round: int | None,
    n_targets: int,
) -> None:
    ax.axis("off")
    ax.set_title("Classification benchmark at a glance", pad=7)

    box = FancyBboxPatch(
        (0.02, 0.05),
        0.96,
        0.87,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9,
        edgecolor="#bdbdbd",
        facecolor="#fafafa",
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    rows = [("Targets", str(n_targets)), ("Initialization", initialization.capitalize())]
    if selected_round is not None:
        rows.append(("Acquisition round", str(selected_round)))
    rows.extend(findings.items())

    y = 0.84
    for label, value in rows:
        ax.text(
            0.07, y, f"{label}:",
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="semibold",
            va="top",
        )
        ax.text(
            0.57, y, str(value),
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
        )
        y -= 0.125


def load_all_inputs(args: argparse.Namespace):
    metadata_rows = []
    descriptor_frames = []
    result_frames = []
    rounds_used = {}

    for target in args.targets:
        target_dir = find_target_dir(args.target_root, target)
        metadata_rows.append(load_metadata(target_dir, target))
        descriptor_frames.append(load_descriptors(target_dir, target))

        summary_file = find_summary_file(args.results_dir, target)
        standardized = standardize_summary(
            summary_file,
            target,
            initialization=args.initialization,
            selected_round=args.round,
        )
        result_frames.append(standardized)

        round_values = standardized["round"].dropna().unique()
        rounds_used[target] = int(round_values[0]) if len(round_values) else None

    metadata = pd.DataFrame(metadata_rows)
    metadata["Target"] = pd.Categorical(
        metadata["Target"], categories=args.targets, ordered=True
    )
    metadata = metadata.sort_values("Target").reset_index(drop=True)
    metadata["Target"] = metadata["Target"].astype(str)

    descriptors, descriptor_mapping = build_descriptor_frame(descriptor_frames)
    results = pd.concat(result_frames, ignore_index=True)
    return metadata, descriptors, descriptor_mapping, results, rounds_used


def build_figure(
    metadata: pd.DataFrame,
    descriptors: pd.DataFrame,
    descriptor_mapping: dict[str, str],
    results: pd.DataFrame,
    targets: list[str],
    initialization: str,
    rounds_used: dict[str, int | None],
) -> plt.Figure:
    figure = plt.figure(figsize=(16.5, 11.2))
    grid = GridSpec(
        nrows=3,
        ncols=14,
        figure=figure,
        height_ratios=[0.95, 1.18, 1.12],
        hspace=0.62,
        wspace=0.42,
    )

    figure.suptitle(
        "Multi-target evaluation of active-learning acquisition strategies "
        "for the classification model",
        y=0.985,
    )

    # A–C: benchmark composition and compact chemical-diversity summary.
    ax_table = figure.add_subplot(grid[0, 0:5])
    add_panel_label(ax_table, "A")
    plot_dataset_table(ax_table, metadata)

    ax_mw = figure.add_subplot(grid[0, 5:10])
    add_panel_label(ax_mw, "B")
    plot_violin(
        ax_mw,
        descriptors,
        targets,
        descriptor_mapping["MolWt"],
        "Molecular weight\n(Da)",
    )

    ax_tpsa = figure.add_subplot(grid[0, 10:14])
    add_panel_label(ax_tpsa, "C")
    plot_violin(
        ax_tpsa,
        descriptors,
        targets,
        descriptor_mapping["TPSA"],
        "TPSA\n(Å²)",
    )

    # D–E: direct comparison of acquisition strategies.
    auc_matrix = (
        results.pivot(index="target", columns="strategy", values="roc_auc_mean")
        .reindex(index=targets, columns=STRATEGY_ORDER)
    )
    actives_matrix = (
        results.pivot(index="target", columns="strategy", values="actives_mean")
        .reindex(index=targets, columns=STRATEGY_ORDER)
    )

    # Leave columns 6–8 empty to separate the two heatmaps and colorbars.
    ax_auc = figure.add_subplot(grid[1, 0:6])
    add_panel_label(ax_auc, "D")
    annotated_heatmap(
        ax_auc,
        auc_matrix,
        title="Predictive performance across targets",
        value_format=".2f",
        colorbar_label="Mean ROC-AUC",
    )

    ax_actives = figure.add_subplot(grid[1, 9:14])
    add_panel_label(ax_actives, "E")
    annotated_heatmap(
        ax_actives,
        actives_matrix,
        title="Active compounds discovered",
        value_format=".1f",
        colorbar_label="Mean actives discovered",
    )

    # F–G: overall ranking and relationship between the two objectives.
    ranked, rank_summary = calculate_ranks(results)

    ax_rank = figure.add_subplot(grid[2, 0:4])
    add_panel_label(ax_rank, "F")
    plot_average_rank(ax_rank, rank_summary)

    ax_scatter = figure.add_subplot(grid[2, 4:14])
    add_panel_label(ax_scatter, "G")
    plot_auc_vs_actives(ax_scatter, results)

    figure.subplots_adjust(
        left=0.055,
        right=0.975,
        top=0.90,
        bottom=0.07,
        hspace=0.62,
        wspace=0.42,
    )

    return figure


def main() -> None:
    args = parse_args()
    set_style()

    metadata, descriptors, descriptor_mapping, results, rounds_used = load_all_inputs(args)

    figure = build_figure(
        metadata=metadata,
        descriptors=descriptors,
        descriptor_mapping=descriptor_mapping,
        results=results,
        targets=args.targets,
        initialization=args.initialization,
        rounds_used=rounds_used,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "figure6_multitarget_classification.png"
    pdf_path = args.output_dir / "figure6_multitarget_classification.pdf"
    svg_path = args.output_dir / "figure6_multitarget_classification.svg"

    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()