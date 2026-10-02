#!/usr/bin/env python3
"""Create exploratory trajectory figures for the frozen v0.5.1 benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "results/representation_benchmark/v051"
ANALYSIS = BENCHMARK / "analysis"
OUTPUT = ANALYSIS / "figures"

REPRESENTATIONS = ["morgan", "rdkit_descriptors", "smi_ted_light", "unimol_v1"]
REPRESENTATION_LABELS = {
    "morgan": "Morgan",
    "rdkit_descriptors": "RDKit descriptors",
    "smi_ted_light": "SMI-TED Light",
    "unimol_v1": "Uni-Mol v1",
}
REPRESENTATION_COLORS = {
    "morgan": "#0072B2",
    "rdkit_descriptors": "#D55E00",
    "smi_ted_light": "#009E73",
    "unimol_v1": "#CC79A7",
}
MODELS = ["random_forest", "bayesian_ridge", "gaussian_process"]
MODEL_LABELS = {
    "random_forest": "Random forest",
    "bayesian_ridge": "Bayesian ridge",
    "gaussian_process": "Gaussian process",
}
MODEL_MARKERS = {
    "random_forest": "o", "bayesian_ridge": "^", "gaussian_process": "s"
}
ACQUISITIONS = ["ucb", "ucb_qed", "ucb_diversity", "ucb_qed_diversity"]
ACQUISITION_LABELS = {
    "ucb": "UCB",
    "ucb_qed": "UCB + QED",
    "ucb_diversity": "UCB + diversity",
    "ucb_qed_diversity": "UCB + QED + diversity",
}
ENDPOINTS = {
    "rmse": ("RMSE", "figure_trajectory_rmse"),
    "joint_discovery_count": (
        "Cumulative affinity + QED discoveries", "figure_trajectory_joint_discovery"
    ),
    "joint_scaffold_count": (
        "Distinct scaffolds among joint discoveries", "figure_trajectory_joint_scaffolds"
    ),
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.2, "axes.labelsize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "figure.titlesize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "grid.linewidth": 0.45,
        "savefig.dpi": 360, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def validate_history(history: pd.DataFrame) -> None:
    expected_rows = 4 * 3 * 4 * 20 * 21
    if len(history) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} history rows, found {len(history)}")
    expected = {
        "representation": set(REPRESENTATIONS), "model": set(MODELS),
        "strategy": set(ACQUISITIONS), "seed": set(range(20)),
        "round": set(range(21)),
    }
    for column, values in expected.items():
        if set(history[column].unique()) != values:
            raise RuntimeError(f"Unexpected {column} values")
    keys = ["representation", "model", "strategy", "seed", "round"]
    if history.duplicated(keys).any():
        raise RuntimeError("Duplicate benchmark trajectory rows")
    if history[list(ENDPOINTS)].isna().any().any():
        raise RuntimeError("Primary plotting endpoint contains unexpected missing values")


def save_figure(fig: plt.Figure, basename: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{basename}.png", bbox_inches="tight", dpi=360)
    fig.savefig(OUTPUT / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def representation_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=REPRESENTATION_COLORS[rep], lw=2,
               label=REPRESENTATION_LABELS[rep])
        for rep in REPRESENTATIONS
    ]


def trajectory_data(history: pd.DataFrame) -> pd.DataFrame:
    groups = ["representation", "model", "strategy", "round"]
    parts = []
    for metric in ENDPOINTS:
        summary = history.groupby(groups, observed=True)[metric].agg(
            n="count", mean="mean", std="std"
        ).reset_index()
        summary["metric"] = metric
        summary["sem"] = summary["std"] / np.sqrt(summary["n"])
        critical = t.ppf(0.975, summary["n"] - 1)
        summary["ci95_half_width"] = critical * summary["sem"]
        summary["ci95_lower"] = summary["mean"] - summary["ci95_half_width"]
        summary["ci95_upper"] = summary["mean"] + summary["ci95_half_width"]
        parts.append(summary)
    result = pd.concat(parts, ignore_index=True)
    expected = len(ENDPOINTS) * 4 * 3 * 4 * 21
    if len(result) != expected or set(result.n) != {20}:
        raise RuntimeError("Trajectory summary is incomplete or does not use 20 seeds")
    return result


def plot_trajectory_figures(data: pd.DataFrame) -> None:
    for metric, (ylabel, basename) in ENDPOINTS.items():
        subset = data[data.metric == metric]
        fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.2), sharex=True, sharey=True)
        for row, model in enumerate(MODELS):
            for column, acquisition in enumerate(ACQUISITIONS):
                ax = axes[row, column]
                panel = subset[(subset.model == model) & (subset.strategy == acquisition)]
                for representation in REPRESENTATIONS:
                    line = panel[panel.representation == representation].sort_values("round")
                    x = line["round"].to_numpy()
                    mean = line["mean"].to_numpy()
                    lower = line["ci95_lower"].to_numpy()
                    upper = line["ci95_upper"].to_numpy()
                    color = REPRESENTATION_COLORS[representation]
                    ax.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0)
                    ax.plot(x, mean, color=color, lw=1.6)
                ax.grid(axis="y", color="#D8D8D8", alpha=0.7)
                ax.set_xticks([0, 5, 10, 15, 20])
                if row == 0:
                    ax.set_title(ACQUISITION_LABELS[acquisition])
                if column == 0:
                    short_label = {
                        "rmse": "RMSE",
                        "joint_discovery_count": "Joint discoveries",
                        "joint_scaffold_count": "Joint-hit scaffolds",
                    }[metric]
                    ax.set_ylabel(short_label)
                    ax.text(
                        0.02, 0.95, MODEL_LABELS[model], transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.5, fontweight="bold",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7,
                              "pad": 1.5},
                    )
                if row == len(MODELS) - 1:
                    ax.set_xlabel("Active-learning round")
        if metric != "rmse":
            # Set the shared limit only after all panels are drawn. Setting it
            # panel-by-panel can lock the shared axis to the first panel range.
            axes[0, 0].set_ylim(0, float(subset["ci95_upper"].max()) * 1.03)
        fig.legend(handles=representation_legend(), loc="upper center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"{ylabel}: representation trajectories (mean and descriptive 95% CI)",
                     y=1.025)
        fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=0.8)
        save_figure(fig, basename)


def paired_difference_data(history: pd.DataFrame) -> pd.DataFrame:
    final = history[history["round"] == 20]
    reference = final[final.representation == "morgan"].set_index(
        ["model", "strategy", "seed"]
    )
    rows = []
    for representation in REPRESENTATIONS[1:]:
        comparison = final[final.representation == representation].set_index(
            ["model", "strategy", "seed"]
        )
        if not comparison.index.equals(reference.index):
            raise RuntimeError(f"Unpaired final-round seeds for {representation}")
        for metric in ENDPOINTS:
            difference = comparison[metric] - reference[metric]
            for (model, strategy, seed), value in difference.items():
                rows.append({
                    "representation": representation, "reference": "morgan",
                    "model": model, "strategy": strategy, "seed": seed,
                    "metric": metric, "difference": value,
                })
    result = pd.DataFrame(rows)
    expected = 3 * len(ENDPOINTS) * 3 * 4 * 20
    if len(result) != expected or result.duplicated(
        ["representation", "model", "strategy", "seed", "metric"]
    ).any():
        raise RuntimeError("Paired final-round figure data are incomplete")
    return result


def plot_paired_differences(data: pd.DataFrame) -> None:
    positions = np.arange(3)
    labels = ["RDKit − Morgan", "SMI-TED − Morgan", "Uni-Mol − Morgan"]
    for metric, (endpoint_label, _) in ENDPOINTS.items():
        subset = data[data.metric == metric]
        fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.2), sharex=True, sharey=True)
        for row, model in enumerate(MODELS):
            for column, acquisition in enumerate(ACQUISITIONS):
                ax = axes[row, column]
                panel = subset[(subset.model == model) & (subset.strategy == acquisition)]
                for position, representation in zip(positions, REPRESENTATIONS[1:]):
                    values = panel[panel.representation == representation].sort_values("seed")
                    # Fixed offsets make every paired seed visible without stochastic jitter.
                    offsets = np.linspace(-0.16, 0.16, 20)
                    ax.scatter(position + offsets, values.difference, s=15,
                               color=REPRESENTATION_COLORS[representation], alpha=0.72,
                               edgecolor="white", linewidth=0.25)
                    ax.plot(position, values.difference.mean(), marker="D", ms=4.2,
                            color="black", zorder=4)
                ax.axhline(0, color="#222222", lw=1.0, zorder=0)
                ax.grid(axis="y", color="#D8D8D8", alpha=0.65)
                if row == 0:
                    ax.set_title(ACQUISITION_LABELS[acquisition])
                if column == 0:
                    ax.set_ylabel(f"{MODEL_LABELS[model]}\nDifference at round 20")
                ax.set_xticks(positions)
                if row == len(MODELS) - 1:
                    ax.set_xticklabels(labels, rotation=25, ha="right")
                else:
                    ax.set_xticklabels([])
        direction = (
            "Negative = lower RMSE than Morgan"
            if metric == "rmse"
            else "Positive = more than Morgan"
        )
        fig.suptitle(
            f"{endpoint_label}: paired seed differences relative to Morgan\n"
            f"All 20 seeds shown; black diamond = mean. {direction}", y=1.035
        )
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.0, w_pad=0.8)
        save_figure(fig, f"figure_paired_difference_{'joint_scaffolds' if metric == 'joint_scaffold_count' else 'joint_discovery' if metric == 'joint_discovery_count' else metric}")


def tradeoff_data(history: pd.DataFrame) -> pd.DataFrame:
    final = history[(history["round"] == 20) & history.strategy.isin(
        ["ucb_qed", "ucb_qed_diversity"]
    )]
    index = ["representation", "model", "seed"]
    discovery = final.pivot(index=index, columns="strategy", values="joint_discovery_count")
    scaffolds = final.pivot(index=index, columns="strategy", values="joint_scaffold_count")
    expected_index = pd.MultiIndex.from_product(
        [REPRESENTATIONS, MODELS, range(20)], names=index
    )
    discovery = discovery.reindex(expected_index)
    scaffolds = scaffolds.reindex(expected_index)
    if discovery.isna().any().any() or scaffolds.isna().any().any():
        raise RuntimeError("QED/diversity discovery pairs are incomplete")
    result = discovery.reset_index()[index].copy()
    for acquisition in ("ucb_qed", "ucb_qed_diversity"):
        result[f"{acquisition}_joint_discovery_count"] = discovery[acquisition].to_numpy()
        result[f"{acquisition}_joint_scaffold_count"] = scaffolds[acquisition].to_numpy()
    return result


def plot_tradeoff(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.2), sharex=True, sharey=True)
    for row, model in enumerate(MODELS):
        for column, representation in enumerate(REPRESENTATIONS):
            ax = axes[row, column]
            panel = data[(data.model == model) & (data.representation == representation)]
            color = REPRESENTATION_COLORS[representation]
            for point in panel.itertuples(index=False):
                x = [point.ucb_qed_joint_discovery_count,
                     point.ucb_qed_diversity_joint_discovery_count]
                y = [point.ucb_qed_joint_scaffold_count,
                     point.ucb_qed_diversity_joint_scaffold_count]
                ax.plot(x, y, color=color, alpha=0.22, lw=0.7)
                ax.scatter(x[0], y[0], s=17, facecolor="white", edgecolor=color,
                           linewidth=0.8, zorder=3)
                ax.scatter(x[1], y[1], s=17, facecolor=color, edgecolor=color,
                           linewidth=0.5, zorder=3)
            ax.grid(color="#D8D8D8", alpha=0.6)
            if row == 0:
                ax.set_title(REPRESENTATION_LABELS[representation])
            if column == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nDistinct joint-hit scaffolds")
            if row == len(MODELS) - 1:
                ax.set_xlabel("Joint discoveries")
    handles = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white",
               markeredgecolor="#555555", lw=0, label="UCB + QED"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555",
               lw=0, label="UCB + QED + diversity"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Paired discovery–scaffold changes after adding diversity", y=1.025)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=0.8)
    save_figure(fig, "figure_qed_diversity_tradeoff")


def prediction_discovery_data(history: pd.DataFrame) -> pd.DataFrame:
    columns = ["representation", "model", "strategy", "seed", "rmse",
               "joint_discovery_count"]
    result = history.loc[history["round"] == 20, columns].copy()
    if len(result) != 960 or result[columns].isna().any().any():
        raise RuntimeError("Final prediction/discovery scatter data are incomplete")
    return result


def plot_prediction_discovery(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.65), sharex=True, sharey=True)
    for column, acquisition in enumerate(ACQUISITIONS):
        ax = axes[column]
        panel = data[data.strategy == acquisition]
        for model in MODELS:
            for representation in REPRESENTATIONS:
                points = panel[(panel.model == model) &
                               (panel.representation == representation)]
                ax.scatter(points.rmse, points.joint_discovery_count, s=21,
                           marker=MODEL_MARKERS[model],
                           color=REPRESENTATION_COLORS[representation], alpha=0.65,
                           edgecolor="white", linewidth=0.25)
        ax.set_title(ACQUISITION_LABELS[acquisition])
        ax.set_xlabel("Final-round RMSE")
        ax.grid(color="#D8D8D8", alpha=0.6)
    axes[0].set_ylabel("Final joint discoveries")
    rep_handles = representation_legend()
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[model], color="#555555", lw=0,
               markerfacecolor="#777777", markersize=6, label=MODEL_LABELS[model])
        for model in MODELS
    ]
    first = fig.legend(handles=rep_handles, loc="upper center", ncol=4,
                       frameon=False, bbox_to_anchor=(0.38, 1.02))
    fig.add_artist(first)
    fig.legend(handles=model_handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.81, 1.02))
    fig.suptitle("Final predictive error and joint discovery across individual campaigns",
                 y=1.11)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    save_figure(fig, "figure_prediction_vs_joint_discovery")


def main() -> None:
    configure_style()
    history = pd.read_parquet(BENCHMARK / "combined_campaign_history.parquet")
    validate_history(history)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    trajectories = trajectory_data(history)
    trajectories.to_csv(OUTPUT / "figure_data_trajectories.csv", index=False)
    plot_trajectory_figures(trajectories)

    differences = paired_difference_data(history)
    differences.to_csv(OUTPUT / "figure_data_paired_differences.csv", index=False)
    plot_paired_differences(differences)

    tradeoff = tradeoff_data(history)
    tradeoff.to_csv(OUTPUT / "figure_data_qed_diversity_tradeoff.csv", index=False)
    plot_tradeoff(tradeoff)

    scatter = prediction_discovery_data(history)
    scatter.to_csv(OUTPUT / "figure_data_prediction_vs_joint_discovery.csv", index=False)
    plot_prediction_discovery(scatter)

    print(
        f"Validated {len(history)} history rows; wrote 8 PNG/PDF figure pairs and "
        f"four reconstruction tables to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
