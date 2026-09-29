#!/usr/bin/env python3
"""Analyze existing v0.5.0 EGFR benchmark artifacts without rerunning them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.regression.v050_analysis import (
    COMPARISONS, FINAL_METRICS, MODELS, STRATEGIES, add_recalculated_endpoints,
    campaign_endpoints, load_artifacts, paired_comparisons, summarize_complete,
    validate, validate_endpoint_consistency,
)

MODEL_LABEL = {m: m.replace("_", " ").title() for m in MODELS}
STRATEGY_LABEL = {s: s.replace("_", " ").upper() for s in STRATEGIES}
COLORS = dict(zip(STRATEGIES, ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]))
DEFAULT_INPUT = Path("results/regression/benchmarks/v050_egfr")
DEFAULT_OUTPUT = Path("results/regression/analysis/v050_egfr")


def save(fig, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all(history: pd.DataFrame, endpoints: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    complete = endpoints[endpoints["complete_campaign"]]
    complete_keys = set(map(tuple, complete[["strategy", "model", "seed"]].itertuples(index=False, name=None)))
    curve = history[[tuple(x) in complete_keys for x in history[["strategy", "model", "seed"]].itertuples(index=False, name=None)]]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})

    def grouped(metric, ylabel, name):
        fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.5), sharey=True)
        for ax, model in zip(axes, MODELS):
            data = complete[complete.model == model]
            for i, strategy in enumerate(STRATEGIES):
                vals = data.loc[data.strategy == strategy, metric].astype(float)
                if len(vals):
                    ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="o", capsize=3,
                                color=COLORS[strategy])
                    ax.scatter(np.full(len(vals), i), vals, s=8, alpha=.25, color=COLORS[strategy])
            ax.set_title(MODEL_LABEL[model] + ("\n(no complete campaigns)" if data.empty else ""))
            ax.set_xticks(range(4), [STRATEGY_LABEL[s] for s in STRATEGIES], rotation=45, ha="right")
        axes[0].set_ylabel(ylabel)
        fig.suptitle("Points: mean ± sample SD; faint points: matched seeds")
        save(fig, output, name)

    grouped("joint_discovery_count", "Joint discoveries after 200 acquisitions (count)", "final_joint_discovery")
    grouped("joint_scaffold_count", "Distinct Bemis–Murcko scaffolds among joint discoveries", "qualifying_scaffolds")

    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.4), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        data = curve[curve.model == model]
        for strategy in STRATEGIES:
            sub = data[data.strategy == strategy]
            if sub.empty: continue
            stats = sub.groupby("n_cumulative_acquired").joint_discovery_count.agg(["mean", "std"]).reset_index()
            ax.plot(stats.n_cumulative_acquired, stats["mean"], color=COLORS[strategy], label=STRATEGY_LABEL[strategy])
            ax.fill_between(stats.n_cumulative_acquired, stats["mean"]-stats["std"], stats["mean"]+stats["std"], color=COLORS[strategy], alpha=.15)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("Newly acquired molecules (count)")
    axes[0].set_ylabel("Cumulative joint discoveries (count)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.25)
    save(fig, output, "joint_discovery_trajectories")

    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.4), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        for strategy in STRATEGIES:
            sub = complete[(complete.model == model) & (complete.strategy == strategy)]
            ax.scatter(sub.affinity_discovery_count, sub.joint_discovery_count, s=18,
                       alpha=.65, color=COLORS[strategy], label=STRATEGY_LABEL[strategy])
        ax.set_title(MODEL_LABEL[model]); ax.set_xlabel("Affinity-qualified discoveries (≤500 nM)")
    axes[0].set_ylabel("Joint discoveries (≤500 nM and QED ≥0.6)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False); fig.subplots_adjust(bottom=.23)
    save(fig, output, "affinity_joint_tradeoff")

    selected = paired[(paired.metric == "joint_discovery_count") & paired.comparison.isin(COMPARISONS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    categories = ["adding_qed_without_diversity", "adding_qed_with_diversity",
                  "adding_diversity_without_qed", "adding_diversity_with_qed"]
    for ax, prefix in zip(axes, ["adding_qed", "adding_diversity"]):
        cats = [c for c in categories if c.startswith(prefix)]
        for mi, model in enumerate(MODELS):
            for ci, comparison in enumerate(cats):
                vals = selected[(selected.model == model) & (selected.comparison == comparison)].paired_difference
                if len(vals):
                    x = mi + (ci-.5)*.22
                    ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt="o", capsize=3)
                    ax.scatter(np.full(len(vals), x), vals, s=7, alpha=.2)
        ax.axhline(0, color="black", lw=.8); ax.set_title(prefix.replace("_", " ").title())
        ax.set_xticks(range(4), [MODEL_LABEL[m] for m in MODELS], rotation=35, ha="right")
    axes[0].set_ylabel("Paired difference in final joint discoveries (count)")
    save(fig, output, "paired_strategy_differences")

    fig, axes = plt.subplots(2, len(MODELS), figsize=(13, 6), sharex="col")
    for col, model in enumerate(MODELS):
        for row, (metric, ylabel) in enumerate([("rmse", "Held-out RMSE (log-affinity units)"),
                                                ("pearson", "Held-out Pearson correlation")]):
            ax = axes[row, col]; data = complete[complete.model == model]
            for i, strategy in enumerate(STRATEGIES):
                vals = data.loc[data.strategy == strategy, metric]
                if len(vals): ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="o", capsize=3, color=COLORS[strategy])
            if row == 0: ax.set_title(MODEL_LABEL[model])
            ax.set_xticks(range(4), [STRATEGY_LABEL[s] for s in STRATEGIES], rotation=45, ha="right")
            if col == 0: ax.set_ylabel(ylabel)
    save(fig, output, "final_prediction_metrics")


def report(output: Path, validation: pd.DataFrame, endpoints: pd.DataFrame,
           summary: pd.DataFrame, consistency: pd.DataFrame) -> None:
    counts = validation.status.value_counts().to_dict()
    complete = endpoints[endpoints.complete_campaign]
    lines = ["# v0.5.0 EGFR benchmark analysis", "",
             "## Validation", "",
             f"Checks: {counts.get('passed',0)} passed, {counts.get('failed',0)} failed, "
             f"{counts.get('unverifiable',0)} unverifiable.", ""]
    for row in validation.itertuples(index=False):
        lines.append(f"- **{row.status.upper()} — {row.check}:** {row.details}")
    lines += ["", "## Scientific results", "",
              "All numerical summaries and figures exclude incomplete campaigns. Error bars are descriptive sample standard deviations across the 20 matched seeds; no inferential interval or hypothesis test is implied.", ""]
    if not summary.empty:
        joint = summary.sort_values("joint_discovery_count_mean", ascending=False)
        for row in joint.itertuples(index=False):
            lines.append(f"- {MODEL_LABEL[row.model]}, {STRATEGY_LABEL[row.strategy]}: "
                         f"final joint discoveries {row.joint_discovery_count_mean:.2f} ± {row.joint_discovery_count_sd:.2f}; "
                         f"affinity-only {row.affinity_discovery_count_mean:.2f} ± {row.affinity_discovery_count_sd:.2f}; "
                         f"qualifying scaffolds {row.joint_scaffold_count_mean:.2f} ± {row.joint_scaffold_count_sd:.2f}.")
    n_complete = int(endpoints.complete_campaign.sum())
    n_campaigns = len(endpoints)
    if n_complete == n_campaigns:
        completeness = (
            f"All {n_campaigns} supplied campaigns are complete; no model or "
            "strategy is excluded from full-budget summaries because of "
            "incomplete artifacts."
        )
    else:
        completeness = (
            f"{n_complete}/{n_campaigns} supplied campaigns are complete. "
            f"The {n_campaigns - n_complete} incomplete campaigns are listed "
            "in incomplete_campaigns.csv and excluded from full-budget "
            "summaries and figures."
        )
    lines += ["", "No composite score or cross-metric ranking was used.", "",
              "## Limitations", "",
              completeness, "",
              "Acquisition-pool and held-out-test identities were not serialized, preventing direct matching and leakage checks. Stored cumulative diversity is reported as generated; per-batch Morgan radius-2/2048-bit diversity is recalculated only as a transparent auxiliary check because the saved rows do not serialize cumulative fingerprints.", "",
              f"Endpoint consistency: {int(consistency.cumulative_counts_agree.sum())}/{len(consistency)} campaigns have saved counts reproduced from per-batch affinities/QED; "
              f"{int(consistency.auc_agrees.sum())}/{len(consistency)} have reproduced AUCs; "
              f"{int(consistency.final_count_agrees_with_curve.sum())}/{len(consistency)} final counts agree with the last cumulative-curve point.", ""]
    (output / "scientific_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "figures").mkdir(exist_ok=True)
    loaded = load_artifacts(args.input)
    history = add_recalculated_endpoints(loaded.history)
    endpoints = campaign_endpoints(history)
    validation = validate(history, loaded.overlaps)
    consistency = validate_endpoint_consistency(history)
    if not consistency[["cumulative_counts_agree", "auc_agrees", "final_count_agrees_with_curve"]].all().all():
        validation.loc[len(validation)] = ["discovery endpoint consistency", "failed", "see endpoint_consistency.csv"]
    else:
        validation.loc[len(validation)] = ["discovery endpoint consistency", "passed", "all saved counts/AUCs reproduce from saved molecule-level batches"]
    summary = summarize_complete(endpoints)
    paired, paired_summary = paired_comparisons(endpoints)
    incomplete = endpoints.loc[~endpoints.complete_campaign, ["strategy", "model", "seed", "observed_rounds", "n_round_rows", "source_directory"]]
    outputs = {"combined_history.csv": history, "campaign_endpoints.csv": endpoints,
               "source_inventory.csv": loaded.sources, "source_overlaps.csv": loaded.overlaps,
               "validation_report.csv": validation, "endpoint_consistency.csv": consistency,
               "incomplete_campaigns.csv": incomplete, "summary_complete_campaigns.csv": summary,
               "paired_comparisons_seed_level.csv": paired, "paired_comparisons_summary.csv": paired_summary}
    for name, frame in outputs.items(): frame.to_csv(args.output / name, index=False)
    plot_all(history, endpoints, paired, args.output)
    report(args.output, validation, endpoints, summary, consistency)
    manifest = {"input": str(args.input), "output": str(args.output),
                "n_history_rows": len(history), "n_campaigns": len(endpoints),
                "n_complete_campaigns": int(endpoints.complete_campaign.sum()),
                "files": sorted(str(p.relative_to(args.output)) for p in args.output.rglob("*") if p.is_file())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
