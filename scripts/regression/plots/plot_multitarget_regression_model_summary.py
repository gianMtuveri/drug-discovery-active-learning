#!/usr/bin/env python3
"""
Create a 3x3 multi-target regression surrogate-model benchmark panel.

Uncertainty-aware models use UCB with beta fixed to 2:
    random_forest, extra_trees, bayesian_ridge, gaussian_process

Deterministic models use greedy acquisition. If no explicit greedy benchmark is
available, a UCB beta=0 run is accepted as the greedy-equivalent fallback:
    gradient_boosting, hist_gradient_boosting, knn, linear_regression

The script scans recursively under results/regression/benchmarks for benchmark
folders containing config.json, final_round_summary.csv and round_summary.csv.
It uses the same ten targets as the classification benchmark.

Outputs PNG, PDF and SVG to results/figures/regression.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path("results/regression/benchmarks")
OUTPUT_DIR = Path("results/figures/regression")
OUTPUT_NAME = "figure9_multitarget_regression_model_summary"

TARGETS = ["EGFR", "JAK2", "PARP1", "BRAF", "ABL1", "SRC", "VEGFR2", "CDK2", "DRD2", "CA2"]
MODELS = [
    "random_forest", "extra_trees", "bayesian_ridge", "gaussian_process",
    "gradient_boosting", "hist_gradient_boosting", "knn", "linear_regression",
]
UCB_MODELS = {"random_forest", "extra_trees", "bayesian_ridge", "gaussian_process"}
DETERMINISTIC_MODELS = set(MODELS) - UCB_MODELS
UCB_BETA = 2.0
GREEDY_BETA_FALLBACK = 0.0

MODEL_LABELS = {
    "random_forest": "RF", "extra_trees": "ET", "bayesian_ridge": "BR",
    "gaussian_process": "GP", "gradient_boosting": "GB",
    "hist_gradient_boosting": "HGB", "knn": "KNN", "linear_regression": "LR",
}
MODEL_LONG_LABELS = {
    "random_forest": "Random forest", "extra_trees": "Extra trees",
    "bayesian_ridge": "Bayesian ridge", "gaussian_process": "Gaussian process",
    "gradient_boosting": "Gradient boosting",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "knn": "K-nearest neighbours", "linear_regression": "Linear regression",
}
MODEL_COLORS = {
    "random_forest": "#4C72B0", "extra_trees": "#55A868",
    "bayesian_ridge": "#C44E52", "gaussian_process": "#8172B2",
    "gradient_boosting": "#CCB974", "hist_gradient_boosting": "#64B5CD",
    "knn": "#8C8C8C", "linear_regression": "#E17C05",
}
MODEL_MARKERS = {
    "random_forest": "o", "extra_trees": "s", "bayesian_ridge": "^",
    "gaussian_process": "D", "gradient_boosting": "v",
    "hist_gradient_boosting": "P", "knn": "X", "linear_regression": "<",
}

@dataclass(frozen=True)
class RunRecord:
    path: Path
    target: str
    strategy: str
    beta: float | None
    final: pd.DataFrame
    rounds: pd.DataFrame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the multi-target regression surrogate-model benchmark.")
    p.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--output-name", default=OUTPUT_NAME)
    p.add_argument("--dataset-summary", type=Path, default=None,
                   help="Optional CSV with target,molecules,median,p25,p75 for panel I.")
    p.add_argument("--round", type=int, default=None, dest="final_round")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--figsize", nargs=2, type=float, default=(17.5, 13.0))
    return p.parse_args()


def normalize_target(value: Any) -> str:
    return str(value).strip().upper()


def load_run(directory: Path) -> RunRecord | None:
    config_path = directory / "config.json"
    final_path = directory / "final_round_summary.csv"
    round_path = directory / "round_summary.csv"
    if not (config_path.is_file() and final_path.is_file() and round_path.is_file()):
        return None
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        final = pd.read_csv(final_path)
        rounds = pd.read_csv(round_path)
    except (json.JSONDecodeError, pd.errors.ParserError) as exc:
        print(f"Warning: skipping {directory}: {exc}", file=sys.stderr)
        return None
    target = normalize_target(config.get("target", ""))
    strategy = str(config.get("strategy", "")).strip().lower()
    beta_raw = config.get("beta")
    beta = None if beta_raw is None else float(beta_raw)
    if not target or not strategy or "model" not in final.columns or "model" not in rounds.columns:
        return None
    return RunRecord(directory, target, strategy, beta, final, rounds)


def discover_runs(root: Path) -> list[RunRecord]:
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {root}")
    runs = []
    for config_path in root.rglob("config.json"):
        record = load_run(config_path.parent)
        if record is not None:
            runs.append(record)
    if not runs:
        raise FileNotFoundError(f"No complete benchmark folders found under {root}")
    return runs


def has_model(record: RunRecord, model: str) -> bool:
    return model in set(record.final["model"].astype(str))


def select_run(runs: list[RunRecord], target: str, model: str) -> RunRecord:
    candidates = [r for r in runs if r.target == target and has_model(r, model)]
    if model in UCB_MODELS:
        exact = [r for r in candidates if r.strategy == "ucb" and r.beta is not None
                 and math.isclose(r.beta, UCB_BETA, abs_tol=1e-12)]
        required = f"UCB beta={UCB_BETA:g}"
    else:
        exact = [r for r in candidates if r.strategy == "greedy"]
        required = "greedy"
        if not exact:
            exact = [r for r in candidates if r.strategy == "ucb" and r.beta is not None
                     and math.isclose(r.beta, GREEDY_BETA_FALLBACK, abs_tol=1e-12)]
            required = "greedy or UCB beta=0 fallback"
    if not exact:
        available = [f"{r.path}(strategy={r.strategy}, beta={r.beta})" for r in candidates]
        raise FileNotFoundError(
            f"No compatible run for target={target}, model={model}. Required {required}. "
            f"Available: {available or 'none'}"
        )
    exact = sorted(exact, key=lambda r: str(r.path))
    if len(exact) > 1:
        print(f"Warning: multiple runs for {target}/{model}; using {exact[0].path}", file=sys.stderr)
    return exact[0]


def infer_round_column(df: pd.DataFrame) -> str:
    for candidate in ("round", "acquisition_round"):
        if candidate in df.columns:
            return candidate
    raise ValueError("round_summary.csv has no round column")


def determine_final_round(selected: dict[tuple[str, str], RunRecord], requested: int | None) -> int:
    shared = min(int(r.rounds[infer_round_column(r.rounds)].max()) for r in selected.values())
    if requested is None:
        return shared
    if requested > shared:
        raise ValueError(f"Requested round {requested} exceeds shared maximum {shared}")
    return requested


def final_row(record: RunRecord, model: str) -> pd.Series:
    rows = record.final.loc[record.final["model"].astype(str) == model]
    if len(rows) != 1:
        raise ValueError(f"Expected one final row for {record.target}/{model} in {record.path}")
    return rows.iloc[0]


def build_final_table(selected: dict[tuple[str, str], RunRecord]) -> pd.DataFrame:
    output = []
    metrics = ["rmse", "pearson", "top20_mean_discovered"]
    for target in TARGETS:
        for model in MODELS:
            record = selected[(target, model)]
            row = final_row(record, model)
            item = {"target": target, "model": model, "strategy": record.strategy, "beta": record.beta}
            for metric in metrics:
                mean_col, std_col = f"{metric}_mean", f"{metric}_std"
                if mean_col not in row.index:
                    raise ValueError(f"{mean_col} missing in {record.path / 'final_round_summary.csv'}")
                item[mean_col] = float(row[mean_col])
                item[std_col] = float(row[std_col]) if std_col in row.index and pd.notna(row[std_col]) else np.nan
            output.append(item)
    return pd.DataFrame(output)


def build_round_table(selected: dict[tuple[str, str], RunRecord], final_round: int) -> pd.DataFrame:
    frames = []
    required = ["rmse_mean", "rmse_std", "top20_mean_discovered_mean", "top20_mean_discovered_std"]
    for target in TARGETS:
        for model in MODELS:
            record = selected[(target, model)]
            frame = record.rounds.copy()
            frame["model"] = frame["model"].astype(str)
            frame = frame.loc[frame["model"] == model].copy()
            round_col = infer_round_column(frame)
            frame = frame.loc[frame[round_col] <= final_round].rename(columns={round_col: "round"})
            missing = [c for c in required if c not in frame.columns]
            if missing:
                raise ValueError(f"Missing columns in {record.path / 'round_summary.csv'}: {missing}")
            frame["target"] = target
            frames.append(frame[["target", "model", "round", *required]])
    return pd.concat(frames, ignore_index=True)


def pivot_metric(final: pd.DataFrame, metric: str) -> np.ndarray:
    table = final.pivot(index="target", columns="model", values=f"{metric}_mean")
    return table.reindex(index=TARGETS, columns=MODELS).to_numpy(dtype=float)


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, fmt: str) -> None:
    finite = values[np.isfinite(values)]
    midpoint = np.nanmedian(finite) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "NA" if not np.isfinite(value) else format(value, fmt)
            color = "white" if np.isfinite(value) and value < midpoint else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.2,
                    fontweight="bold", color=color)


def plot_heatmap(ax: plt.Axes, values: np.ndarray, title: str, cbar_label: str, fmt: str) -> None:
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(TARGETS)))
    ax.set_yticklabels(TARGETS, fontsize=8)
    annotate_heatmap(ax, values, fmt)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.axvline(3.5, color="white", linewidth=2.5)
    ax.text(1.5, -1.35, "Uncertainty-aware", ha="center", fontsize=8, fontweight="bold")
    ax.text(5.5, -1.35, "Deterministic", ha="center", fontsize=8, fontweight="bold")


def compute_ranks(final: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    frame = final[["target", "model", f"{metric}_mean"]].copy()
    frame["rank"] = frame.groupby("target")[f"{metric}_mean"].rank(
        method="average", ascending=not higher_is_better
    )
    ranked = frame.groupby("model", as_index=False)["rank"].agg(["mean", "std"]).reset_index()
    ranked = ranked.rename(columns={"mean": "rank_mean", "std": "rank_std"})
    return ranked.set_index("model").reindex(MODELS).reset_index()


def plot_rank_bars(ax: plt.Axes, ranks: pd.DataFrame, title: str) -> None:
    y = np.arange(len(MODELS))
    ax.barh(y, ranks["rank_mean"], xerr=ranks["rank_std"],
            color=[MODEL_COLORS[m] for m in MODELS], alpha=0.85,
            error_kw={"elinewidth": 1.0, "capsize": 2.5})
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank across targets\n(lower is better)", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def aggregate_models(final: pd.DataFrame) -> pd.DataFrame:
    grouped = final.groupby("model", as_index=False).agg(
        rmse_mean=("rmse_mean", "mean"),
        rmse_std=("rmse_mean", "std"),
        discovery_mean=("top20_mean_discovered_mean", "mean"),
        discovery_std=("top20_mean_discovered_mean", "std"),
    )
    return grouped.set_index("model").reindex(MODELS).reset_index()


def plot_tradeoff(ax: plt.Axes, data: pd.DataFrame) -> None:
    for _, row in data.iterrows():
        model = str(row["model"])
        ax.errorbar(row["rmse_mean"], row["discovery_mean"],
                    xerr=row["rmse_std"], yerr=row["discovery_std"],
                    fmt=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                    markersize=6, capsize=2.5, elinewidth=1.0)
        ax.annotate(MODEL_LABELS[model], (row["rmse_mean"], row["discovery_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Mean RMSE across targets\n(lower is better)", fontsize=8)
    ax.set_ylabel("Mean top-20 discovered affinity\n(higher is better)", fontsize=8)
    ax.set_title("Prediction-discovery trade-off", fontsize=11, fontweight="bold", pad=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def aggregate_rounds(rounds: pd.DataFrame, metric: str) -> pd.DataFrame:
    return rounds.groupby(["model", "round"], as_index=False).agg(
        mean=(f"{metric}_mean", "mean"),
        between_target_std=(f"{metric}_mean", "std"),
    )


def plot_round_curves(ax: plt.Axes, curves: pd.DataFrame, title: str, ylabel: str) -> None:
    for model in MODELS:
        subset = curves.loc[curves["model"] == model].sort_values("round")
        x = subset["round"].to_numpy(float)
        mean = subset["mean"].to_numpy(float)
        std = subset["between_target_std"].to_numpy(float)
        ax.plot(x, mean, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                markersize=3.5, linewidth=1.5, label=MODEL_LONG_LABELS[model])
        ax.fill_between(x, mean - std, mean + std, color=MODEL_COLORS[model], alpha=0.10)
    ax.set_xlabel("Active-learning round", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_policy_table(ax: plt.Axes) -> None:
    ax.axis("off")
    rows = []
    for model in MODELS:
        if model in UCB_MODELS:
            uncertainty = "Native" if model in {"bayesian_ridge", "gaussian_process"} else "Tree ensemble"
            acquisition = f"UCB (β={UCB_BETA:g})"
        else:
            uncertainty = "Not used"
            acquisition = "Greedy"
        rows.append([MODEL_LABELS[model], uncertainty, acquisition])
    table = ax.table(cellText=rows, colLabels=["Model", "Uncertainty", "Acquisition"],
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F0F0F0")
    ax.set_title("Model and acquisition summary", fontsize=11, fontweight="bold", pad=8)


def plot_dataset_table(ax: plt.Axes, path: Path) -> None:
    data = pd.read_csv(path)
    required = ["target", "molecules", "median", "p25", "p75"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset summary missing columns: {missing}")
    data["target"] = data["target"].map(normalize_target)
    data = data.set_index("target").reindex(TARGETS).reset_index()
    rows = [[r.target, f"{int(r.molecules):,}", f"{r.median:.2f}", f"{r.p25:.2f}", f"{r.p75:.2f}"]
            for r in data.itertuples(index=False)]
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Target", "Molecules", "Median", "P25", "P75"],
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.05, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F0F0F0")
    ax.set_title("Regression dataset summary", fontsize=11, fontweight="bold", pad=8)


def add_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="left", va="top")


def create_figure(final: pd.DataFrame, rounds: pd.DataFrame, final_round: int,
                  dataset_summary: Path | None, figsize: tuple[float, float]) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.82, 0.88], hspace=0.43, wspace=0.34)
    axes = [fig.add_subplot(grid[r, c]) for r in range(3) for c in range(3)]

    plot_heatmap(axes[0], pivot_metric(final, "rmse"), "Prediction error (RMSE)", "RMSE", ".3f")
    plot_heatmap(axes[1], pivot_metric(final, "pearson"), "Prediction correlation", "Pearson correlation", ".3f")
    plot_heatmap(axes[2], pivot_metric(final, "top20_mean_discovered"),
                 "Lead discovery (top-20 mean affinity)", "Mean affinity", ".2f")

    plot_rank_bars(axes[3], compute_ranks(final, "rmse", False), "Prediction-quality ranking")
    plot_rank_bars(axes[4], compute_ranks(final, "top20_mean_discovered", True), "Lead-discovery ranking")
    plot_tradeoff(axes[5], aggregate_models(final))

    plot_round_curves(axes[6], aggregate_rounds(rounds, "rmse"),
                      "Prediction error across active-learning rounds", "RMSE")
    plot_round_curves(axes[7], aggregate_rounds(rounds, "top20_mean_discovered"),
                      "Lead discovery across active-learning rounds", "Top-20 mean affinity")

    if dataset_summary is None:
        plot_policy_table(axes[8])
    else:
        plot_dataset_table(axes[8], dataset_summary)

    for letter, ax in zip("ABCDEFGHI", axes, strict=True):
        add_letter(ax, letter)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                          linewidth=1.6, markersize=5) for m in MODELS]
    labels = [MODEL_LONG_LABELS[m] for m in MODELS]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=4, frameon=False, fontsize=8.5, columnspacing=1.5, handlelength=2.3)
    fig.suptitle(
        "Multi-target regression surrogate-model benchmark\n"
        f"Final comparison at round {final_round}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.075)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, output_name: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{output_name}.{ext}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    try:
        runs = discover_runs(args.benchmark_root.resolve())
        selected = {(target, model): select_run(runs, target, model)
                    for target in TARGETS for model in MODELS}
        final_round = determine_final_round(selected, args.final_round)
        final = build_final_table(selected)
        rounds = build_round_table(selected, final_round)
        fig = create_figure(final, rounds, final_round, args.dataset_summary, tuple(args.figsize))
        saved = save_figure(fig, args.output_dir, args.output_name, args.dpi)
        plt.close(fig)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Used UCB beta={UCB_BETA:g} for uncertainty-aware models.")
    print("Used greedy acquisition for deterministic models.")
    print(f"Final round: {final_round}")
    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
