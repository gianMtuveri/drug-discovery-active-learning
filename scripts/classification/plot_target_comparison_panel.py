import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STRATEGIES = [
    "greedy",
    "random",
    "uncertainty_topk",
    "uncertainty_diverse",
    "query_by_committee",
]

STRATEGY_LABELS = {
    "greedy": "Greedy",
    "random": "Random",
    "uncertainty_topk": "Top-K",
    "uncertainty_diverse": "Diverse",
    "query_by_committee": "QBC",
}

DESCRIPTORS = [
    "MolWt",
    "LogP",
    "TPSA",
    #"HBD",
    #"HBA",
    "RotatableBonds",
    "RingCount",
    "FractionCSP3",
]


def load_summary(target):
    path = Path(f"results/tables/{target.lower()}_classification_summary.csv")
    df = pd.read_csv(path)
    df = df[df["round"] == 10].copy()

    # For now use the best initialization per strategy.
    df = (
        df.sort_values("roc_auc_mean", ascending=False)
        .groupby("strategy", as_index=False)
        .first()
    )

    df["target"] = target
    return df


def load_metadata(target):
    path = Path(f"data/processed/targets/{target}/metadata.json")
    with open(path) as f:
        return json.load(f)


def load_descriptors(target):
    path = Path(f"data/processed/targets/{target}/descriptors.parquet")
    df = pd.read_parquet(path)
    df["target"] = target
    return df


def make_metric_matrix(summary, metric):
    matrix = (
        summary.pivot(index="target", columns="strategy", values=metric)
        .reindex(columns=STRATEGIES)
    )
    return matrix


def plot_heatmap(ax, matrix, title, cbar_label, fmt=".3f"):
    im = ax.imshow(matrix.values, aspect="auto")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        [STRATEGY_LABELS[c] for c in matrix.columns],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)

    vmin = np.nanmin(matrix.values)
    vmax = np.nanmax(matrix.values)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]

            norm_value = (value - vmin) / (vmax - vmin + 1e-12)
            text_color = "white" if norm_value < 0.35 else "black"

            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)


def plot_rank_table(ax, roc_matrix):
    ax.axis("off")
    ax.set_title("Strategy ranking by ROC AUC", fontsize=11, fontweight="bold")

    ranks = []
    for target in roc_matrix.index:
        ordered = roc_matrix.loc[target].sort_values(ascending=False)
        row = [target] + [STRATEGY_LABELS[s] for s in ordered.index]
        ranks.append(row)

    columns = ["Target", "Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]

    table = ax.table(
        cellText=ranks,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.4)
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
        else:
            cell.set_facecolor("white")


def plot_pareto(ax, summary):
    ax.set_title("ROC AUC vs actives discovered", fontsize=11, fontweight="bold")
    ax.set_xlabel("ROC AUC mean")
    ax.set_ylabel("Actives discovered mean")

    for target in summary["target"].unique():
        sub_target = summary[summary["target"] == target]
        ax.plot(
            sub_target["roc_auc_mean"],
            sub_target["actives_mean"],
            alpha=0.25,
            linewidth=0.8,
        )

    for strategy in STRATEGIES:
        sub = summary[summary["strategy"] == strategy]
        ax.scatter(
            sub["roc_auc_mean"],
            sub["actives_mean"],
            label=STRATEGY_LABELS[strategy],
            alpha=0.85,
            s=30,
        )

    ax.legend(fontsize=6, frameon=True, loc="upper right")
    ax.grid(alpha=0.25)


def plot_descriptor_distributions(fig, outer_ax, descriptors):
    outer_ax.axis("off")

    subfig = outer_ax.get_subplotspec().subgridspec(
        2,
        4,
        wspace=0.42,
        hspace=0.85,
    )

    targets = sorted(descriptors["target"].unique())

    descriptor_labels = {
        "MolWt": "Molecular weight\n(Da)",
        "LogP": "LogP\n(XlogP3)",
        "TPSA": "TPSA\n(Å²)",
        #"HBD": "H-bond donors\n(count)",
        #"HBA": "H-bond acceptors\n(count)",
        "RotatableBonds": "Rotatable bonds\n(count)",
        "RingCount": "Ring count\n(count)",
        "FractionCSP3": "Fraction sp³",
    }

    for idx, descriptor in enumerate(DESCRIPTORS):
        ax = fig.add_subplot(subfig[idx // 4, idx % 4])

        data = [
            descriptors.loc[
                descriptors["target"] == target,
                descriptor,
            ].dropna().values
            for target in targets
        ]

        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        for body in parts["bodies"]:
            body.set_alpha(0.70)

        ax.set_title(
            descriptor_labels.get(descriptor, descriptor),
            fontsize=8,
            pad=8,
        )

        ax.set_xticks(np.arange(1, len(targets) + 1))
        ax.set_xticklabels(
            targets,
            rotation=60,
            ha="right",
            fontsize=6,
        )

        ax.grid(alpha=0.20)

        # Robust limits to avoid rare huge molecules compressing the plot.
        if descriptor != "FractionCSP3":
            ymin = descriptors[descriptor].quantile(0.01)
            ymax = descriptors[descriptor].quantile(0.99)
            ax.set_ylim(max(0, ymin), ymax)
        else:
            ax.set_ylim(0, 1)


def plot_descriptor_correlation(ax, descriptors):
    ax.set_title("Descriptor correlation")

    corr = descriptors[DESCRIPTORS].corr()

    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(DESCRIPTORS)))
    ax.set_xticklabels(DESCRIPTORS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(DESCRIPTORS)))
    ax.set_yticklabels(DESCRIPTORS, fontsize=7)

    for i in range(len(DESCRIPTORS)):
        for j in range(len(DESCRIPTORS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_metadata_table(ax, metadata):
    ax.axis("off")
    ax.set_title("Dataset summary")

    rows = []
    for target, meta in metadata.items():
        rows.append(
            [
                target,
                meta["n_molecules_valid"],
                f"{meta['active_fraction']:.3f}",
                meta["activity_threshold_nM"],
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=["Target", "Molecules", "Active frac.", "Threshold nM"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.3)


def plot_win_counts(ax, roc_matrix):
    ax.set_title("Rank-1 strategy counts", fontsize=11, fontweight="bold")

    winners = roc_matrix.idxmax(axis=1)
    counts = winners.value_counts().reindex(STRATEGIES, fill_value=0)

    labels = [STRATEGY_LABELS[s] for s in counts.index]

    ax.bar(labels, counts.values)

    ax.set_ylabel("Number of targets")
    ax.set_ylim(0, counts.max() + 1)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.05, str(value), ha="center", va="bottom")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="results/figures/target_comparison_panel.png",
    )
    args = parser.parse_args()

    summaries = []
    metadata = {}
    descriptor_tables = []

    for target in args.targets:
        summaries.append(load_summary(target))
        metadata[target] = load_metadata(target)
        descriptor_tables.append(load_descriptors(target))

    summary = pd.concat(summaries, ignore_index=True)
    descriptors = pd.concat(descriptor_tables, ignore_index=True)

    roc_matrix = make_metric_matrix(summary, "roc_auc_mean")
    actives_matrix = make_metric_matrix(summary, "actives_mean")

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        height_ratios=[1.0, 1.25, 1.0],
        wspace=0.42,
        hspace=0.60,
    )

    ax = fig.add_subplot(gs[0, 0])
    plot_heatmap(ax, roc_matrix, "ROC AUC heatmap", "ROC AUC", fmt=".3f")

    ax = fig.add_subplot(gs[0, 1])
    plot_heatmap(ax, actives_matrix, "Actives discovered heatmap", "Actives", fmt=".1f")

    ax = fig.add_subplot(gs[0, 2])
    plot_win_counts(ax, roc_matrix)

    ax = fig.add_subplot(gs[0, 3])
    plot_pareto(ax, summary)

    ax = fig.add_subplot(gs[1, :])
    plot_descriptor_distributions(fig, ax, descriptors)

    fig.text(
        0.5,
        0.655,
        "Physicochemical descriptor distributions across targets",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax = fig.add_subplot(gs[2, 0:1])
    plot_descriptor_correlation(ax, descriptors)

    ax = fig.add_subplot(gs[2, 1:3])
    plot_metadata_table(ax, metadata)

    fig.suptitle(
        "Active-learning benchmark across BindingDB targets",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    ax = fig.add_subplot(gs[2, 3:])
    ax.axis("off")
    ax.text(
        0,
        1,
        (
            "Notes\n\n"
            "• Thresholds calibrated per target.\n"
            "• ROC AUC measures model quality.\n"
            "• Actives discovered measures hit-finding.\n"
            "• Greedy maximizes actives but weakens learning.\n"
            "• Diversity-aware uncertainty is most consistent."
        ),
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()