import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGIES = [
    "random",
    "greedy",
    "uncertainty",
    "uncertainty_diverse",
]

STRATEGY_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "uncertainty": "Uncertainty",
    "uncertainty_diverse": "Uncertainty + diversity",
}


def load_target_summary(target: str) -> pd.DataFrame:
    path = Path(
        f"results/tables/"
        f"{target.lower()}_regression_active_learning_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    df = pd.read_csv(path)
    df["target"] = target

    return df


def load_target_affinity(target: str) -> np.ndarray:
    path = Path(
        f"data/processed/targets/{target}/affinity.npy"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing affinity file: {path}")

    return np.load(path)


def make_metric_matrix(
    final_summary: pd.DataFrame,
    metric: str,
    targets: list[str],
) -> pd.DataFrame:
    return (
        final_summary
        .pivot(
            index="target",
            columns="strategy",
            values=metric,
        )
        .reindex(index=targets, columns=STRATEGIES)
    )


def plot_heatmap(
    ax,
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    lower_is_better: bool,
    fmt: str = ".3f",
):
    values = matrix.to_numpy(dtype=float)

    im = ax.imshow(
        values,
        aspect="auto",
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        [STRATEGY_LABELS[strategy] for strategy in matrix.columns],
        rotation=35,
        ha="right",
        fontsize=8,
    )

    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    for row_idx in range(values.shape[0]):
        row = values[row_idx]

        if lower_is_better:
            winner_idx = np.nanargmin(row)
        else:
            winner_idx = np.nanargmax(row)

        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]

            normalized = (
                (value - vmin) / (vmax - vmin + 1e-12)
            )

            text_color = (
                "white"
                if normalized < 0.35
                else "black"
            )

            ax.text(
                col_idx,
                row_idx,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight=(
                    "bold"
                    if col_idx == winner_idx
                    else "normal"
                ),
            )

        ax.add_patch(
            plt.Rectangle(
                (winner_idx - 0.5, row_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )

    colorbar = plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(colorbar_label, fontsize=8)


def plot_win_counts(
    ax,
    matrix: pd.DataFrame,
    title: str,
    lower_is_better: bool,
):
    if lower_is_better:
        winners = matrix.idxmin(axis=1)
    else:
        winners = matrix.idxmax(axis=1)

    counts = (
        winners
        .value_counts()
        .reindex(STRATEGIES, fill_value=0)
    )

    labels = [
        STRATEGY_LABELS[strategy]
        for strategy in STRATEGIES
    ]

    positions = np.arange(len(labels))

    ax.bar(
        positions,
        counts.to_numpy(),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Number of targets")
    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    upper = max(1, int(counts.max()) + 1)
    ax.set_ylim(0, upper)

    for position, value in zip(
        positions,
        counts.to_numpy(),
    ):
        ax.text(
            position,
            value + 0.05,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_tradeoff(
    ax,
    final_summary: pd.DataFrame,
):
    ax.set_title(
        "Predictive performance vs potency discovery",
        fontsize=11,
        fontweight="bold",
    )

    for target in final_summary["target"].unique():
        target_data = final_summary[
            final_summary["target"] == target
        ]

        ax.plot(
            target_data["rmse_mean"],
            target_data["top20_mean_discovered_mean"],
            linewidth=0.7,
            alpha=0.25,
        )

    for strategy in STRATEGIES:
        strategy_data = final_summary[
            final_summary["strategy"] == strategy
        ]

        ax.scatter(
            strategy_data["rmse_mean"],
            strategy_data["top20_mean_discovered_mean"],
            label=STRATEGY_LABELS[strategy],
            s=35,
            alpha=0.8,
        )

    ax.set_xlabel("RMSE — lower is better")
    ax.set_ylabel("Top-20 mean pAffinity — higher is better")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def plot_mean_learning_curve(
    ax,
    full_summary: pd.DataFrame,
    mean_column: str,
    std_column: str,
    title: str,
    ylabel: str,
):
    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    for strategy in STRATEGIES:
        strategy_data = full_summary[
            full_summary["strategy"] == strategy
        ]

        aggregate = (
            strategy_data
            .groupby("round")
            .agg(
                mean=(mean_column, "mean"),
                target_std=(mean_column, "std"),
            )
            .reset_index()
        )

        rounds = aggregate["round"].to_numpy()
        mean = aggregate["mean"].to_numpy()
        target_std = aggregate["target_std"].fillna(0).to_numpy()

        ax.plot(
            rounds,
            mean,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=STRATEGY_LABELS[strategy],
        )

        ax.fill_between(
            rounds,
            mean - target_std,
            mean + target_std,
            alpha=0.12,
        )

    ax.set_xlabel("Active-learning round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def plot_dataset_summary(
    ax,
    targets: list[str],
):
    ax.axis("off")
    ax.set_title(
        "Regression dataset summary",
        fontsize=11,
        fontweight="bold",
    )

    rows = []

    for target in targets:
        affinity = load_target_affinity(target)

        rows.append(
            [
                target,
                f"{len(affinity):,}",
                f"{np.median(affinity):.2f}",
                f"{np.min(affinity):.2f}",
                f"{np.max(affinity):.2f}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Target",
            "Molecules",
            "Median",
            "Min",
            "Max",
        ],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.4)

        if row == 0:
            cell.set_text_props(fontweight="bold")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--round",
        type=int,
        default=10,
        help="Round used for final-performance heatmaps.",
    )

    parser.add_argument(
        "--output",
        default=(
            "results/figures/"
            "regression_target_comparison_panel.png"
        ),
    )

    args = parser.parse_args()

    summaries = [
        load_target_summary(target)
        for target in args.targets
    ]

    full_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    final_summary = full_summary[
        full_summary["round"] == args.round
    ].copy()

    missing_targets = (
        set(args.targets)
        - set(final_summary["target"].unique())
    )

    if missing_targets:
        raise ValueError(
            "No final-round data found for: "
            + ", ".join(sorted(missing_targets))
        )

    rmse_matrix = make_metric_matrix(
        final_summary,
        "rmse_mean",
        args.targets,
    )

    r2_matrix = make_metric_matrix(
        final_summary,
        "r2_mean",
        args.targets,
    )

    top20_matrix = make_metric_matrix(
        final_summary,
        "top20_mean_discovered_mean",
        args.targets,
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 14),
    )

    plot_heatmap(
        axes[0, 0],
        rmse_matrix,
        title="Final RMSE",
        colorbar_label="RMSE",
        lower_is_better=True,
    )

    plot_heatmap(
        axes[0, 1],
        r2_matrix,
        title="Final R²",
        colorbar_label="R²",
        lower_is_better=False,
    )

    plot_heatmap(
        axes[0, 2],
        top20_matrix,
        title="Top-20 discovered potency",
        colorbar_label="Mean pAffinity",
        lower_is_better=False,
    )

    plot_win_counts(
        axes[1, 0],
        rmse_matrix,
        title="Predictive-performance wins",
        lower_is_better=True,
    )

    plot_win_counts(
        axes[1, 1],
        top20_matrix,
        title="Potency-discovery wins",
        lower_is_better=False,
    )

    plot_tradeoff(
        axes[1, 2],
        final_summary,
    )

    plot_mean_learning_curve(
        axes[2, 0],
        full_summary,
        mean_column="rmse_mean",
        std_column="rmse_std",
        title="Mean RMSE across targets",
        ylabel="RMSE",
    )

    plot_mean_learning_curve(
        axes[2, 1],
        full_summary,
        mean_column="top20_mean_discovered_mean",
        std_column="top20_mean_discovered_std",
        title="Mean discovered potency across targets",
        ylabel="Top-20 mean pAffinity",
    )

    plot_dataset_summary(
        axes[2, 2],
        args.targets,
    )

    fig.suptitle(
        (
            "Multi-target regression active-learning benchmark\n"
            f"Final comparison at round {args.round}"
        ),
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95],
        h_pad=2.0,
        w_pad=1.5,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()