#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_INPUT_ROOT = Path("results/regression/benchmarks/multitarget")
DEFAULT_OUTPUT_ROOT = Path("results/regression/analysis/multitarget")

TARGETS = ["EGFR","JAK2","PARP1","BRAF","ABL1","SRC","VEGFR2","CDK2","DRD2","CA2"]
MODELS = [
    "random_forest","extra_trees","bayesian_ridge","gaussian_process",
    "gradient_boosting","hist_gradient_boosting","knn","linear_regression",
]
ORIGINAL_MODELS = [m for m in MODELS if m != "bayesian_ridge"]
UNCERTAINTY_AWARE_MODELS = [
    "random_forest","extra_trees","bayesian_ridge","gaussian_process",
]
FINAL_METRICS = [
    "rmse","mae","r2","pearson",
    "best_discovered","top20_mean_discovered","mean_discovered",
]
HIGHER_IS_BETTER = {
    "rmse": False, "mae": False, "r2": True, "pearson": True,
    "best_discovered": True, "top20_mean_discovered": True, "mean_discovered": True,
}
PREDICTION_METRICS = ["rmse","mae","r2","pearson"]
DISCOVERY_METRICS = ["best_discovered","top20_mean_discovered","mean_discovered"]

@dataclass(frozen=True)
class RunFiles:
    target: str
    run_type: str
    directory: Path
    config: dict
    history_path: Path
    round_summary_path: Path
    final_summary_path: Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--expected-beta", type=float, default=2.0)
    p.add_argument("--expected-seeds", type=int, default=20)
    p.add_argument("--expected-rounds", type=int, default=20)
    return p.parse_args()

def read_config(directory: Path) -> dict:
    with (directory / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def required_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {directory}")
    return path

def discover_target_runs(root: Path, target: str):
    low = target.lower()
    corrected_dir = root / f"{low}_ucb_beta2_bayesian_ridge_corrected"
    if not corrected_dir.is_dir():
        raise FileNotFoundError(f"Missing corrected Bayesian Ridge run: {corrected_dir}")

    pattern = re.compile(rf"^{re.escape(low)}_ucb_\d{{8}}_\d{{6}}$")
    originals = sorted(p for p in root.iterdir() if p.is_dir() and pattern.match(p.name))
    if len(originals) != 1:
        raise ValueError(
            f"{target}: expected one timestamped original run, found "
            f"{len(originals)}: {[p.name for p in originals]}"
        )
    original_dir = originals[0]

    def build(directory: Path, run_type: str):
        return RunFiles(
            target=target,
            run_type=run_type,
            directory=directory,
            config=read_config(directory),
            history_path=required_file(directory, "history.csv"),
            round_summary_path=required_file(directory, "round_summary.csv"),
            final_summary_path=required_file(directory, "final_round_summary.csv"),
        )

    return build(original_dir, "original"), build(corrected_dir, "bayesian_ridge_corrected")

def validate_config(run: RunFiles, beta: float, seeds: int, rounds: int):
    problems = []
    cfg = run.config
    if str(cfg.get("target","")).upper() != run.target:
        problems.append(f"{run.directory.name}: wrong target in config")
    if str(cfg.get("strategy","")).lower() != "ucb":
        problems.append(f"{run.directory.name}: strategy is not ucb")
    if cfg.get("beta") is None or not np.isclose(float(cfg["beta"]), beta):
        problems.append(f"{run.directory.name}: beta={cfg.get('beta')} expected {beta}")
    if cfg.get("seeds") is not None and int(cfg["seeds"]) != seeds:
        problems.append(f"{run.directory.name}: seeds={cfg.get('seeds')} expected {seeds}")
    if cfg.get("rounds") is not None and int(cfg["rounds"]) != rounds:
        problems.append(f"{run.directory.name}: rounds={cfg.get('rounds')} expected {rounds}")

    models = [str(x) for x in cfg.get("models", [])]
    if run.run_type == "bayesian_ridge_corrected":
        if set(models) != {"bayesian_ridge"}:
            problems.append(f"{run.directory.name}: corrected run should contain only bayesian_ridge")
    else:
        missing = sorted(set(ORIGINAL_MODELS) - set(models))
        if missing:
            problems.append(f"{run.directory.name}: missing original models {missing}")
    return problems

def infer_round_col(df):
    for c in ("round","acquisition_round"):
        if c in df.columns:
            return c
    raise ValueError("No round column found")

def infer_seed_col(df):
    for c in ("seed","random_state"):
        if c in df.columns:
            return c
    raise ValueError("No seed column found")

def filter_models(df, run):
    df = df.copy()
    df["model"] = df["model"].astype(str)
    if run.run_type == "bayesian_ridge_corrected":
        return df[df["model"] == "bayesian_ridge"].copy()
    return df[df["model"].isin(ORIGINAL_MODELS)].copy()

def prepare_history(run):
    df = pd.read_csv(run.history_path)
    if "model" not in df.columns:
        raise ValueError(f"{run.history_path}: missing model")
    df = df.rename(columns={infer_seed_col(df): "seed", infer_round_col(df): "round"})
    df = filter_models(df, run)
    df["target"] = run.target
    df["source_path"] = str(run.directory)
    df["source_run_type"] = run.run_type
    df["corrected_run"] = run.run_type == "bayesian_ridge_corrected"
    return df

def prepare_summary(run, filename):
    path = run.round_summary_path if filename == "round_summary.csv" else run.final_summary_path
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"{path}: missing model")
    if filename == "round_summary.csv":
        rc = infer_round_col(df)
        if rc != "round":
            df = df.rename(columns={rc: "round"})
    df = filter_models(df, run)
    df["target"] = run.target
    df["source_path"] = str(run.directory)
    df["source_run_type"] = run.run_type
    df["corrected_run"] = run.run_type == "bayesian_ridge_corrected"
    return df

def validate_history(history, seeds, rounds):
    problems = []
    dup = history.duplicated(["target","model","seed","round"], keep=False)
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicated target/model/seed/round rows")
    expected = {(t,m) for t in TARGETS for m in MODELS}
    observed = set(zip(history["target"], history["model"].astype(str)))
    if expected - observed:
        problems.append(f"Missing target/model pairs: {sorted(expected-observed)}")
    for (target,model), group in history.groupby(["target","model"]):
        if group["seed"].nunique() != seeds:
            problems.append(f"{target}/{model}: {group['seed'].nunique()} seeds, expected {seeds}")
        positive_rounds = pd.to_numeric(group["round"], errors="coerce")
        if positive_rounds[positive_rounds > 0].nunique() < rounds:
            problems.append(f"{target}/{model}: fewer than {rounds} positive rounds")
    return problems

def validate_final(final):
    problems = []
    counts = final.groupby(["target","model"]).size()
    bad = counts[counts != 1]
    if not bad.empty:
        problems.append("Final summary does not have one row per target/model:\n" + bad.to_string())
    for metric in FINAL_METRICS:
        if f"{metric}_mean" not in final.columns:
            problems.append(f"Missing final metric column: {metric}_mean")
    return problems

def make_ranks(final, models):
    subset = final[final["model"].isin(models)].copy()
    out = subset[["target","model"]].drop_duplicates()
    for metric in FINAL_METRICS:
        col = f"{metric}_mean"
        if col not in subset.columns:
            continue
        temp = subset[["target","model",col]].copy()
        temp[f"{metric}_rank"] = temp.groupby("target")[col].rank(
            method="average", ascending=not HIGHER_IS_BETTER[metric]
        )
        out = out.merge(temp[["target","model",f"{metric}_rank"]], on=["target","model"])
    pred_cols = [f"{m}_rank" for m in PREDICTION_METRICS if f"{m}_rank" in out.columns]
    disc_cols = [f"{m}_rank" for m in DISCOVERY_METRICS if f"{m}_rank" in out.columns]
    out["prediction_composite_rank"] = out[pred_cols].mean(axis=1)
    out["discovery_composite_rank"] = out[disc_cols].mean(axis=1)
    out["overall_composite_rank"] = out[pred_cols + disc_cols].mean(axis=1)
    return out.sort_values(["target","model"]).reset_index(drop=True)

def rank_summary(ranks):
    rank_cols = [c for c in ranks.columns if c.endswith("_rank")]
    rows = []
    for model, g in ranks.groupby("model"):
        row = {"model": model, "n_targets": g["target"].nunique()}
        for c in rank_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_median"] = g[c].median()
            row[f"{c}_std"] = g[c].std()
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("overall_composite_rank_mean").reset_index(drop=True)

def winners(final, models):
    subset = final[final["model"].isin(models)]
    rows = []
    for target in TARGETS:
        tdf = subset[subset["target"] == target]
        for metric in FINAL_METRICS:
            col = f"{metric}_mean"
            if col not in tdf.columns:
                continue
            best = tdf[col].max() if HIGHER_IS_BETTER[metric] else tdf[col].min()
            win = tdf[np.isclose(tdf[col].astype(float), float(best))]["model"].astype(str).tolist()
            rows.append({
                "target": target, "metric": metric, "best_value": best,
                "winner_models": "|".join(win), "n_winners": len(win),
            })
    return pd.DataFrame(rows)

def prediction_discovery(final, ranks_all, ranks_uq):
    grouped = final.groupby("model", as_index=False).agg(
        mean_rmse=("rmse_mean","mean"),
        std_rmse_across_targets=("rmse_mean","std"),
        mean_mae=("mae_mean","mean"),
        mean_r2=("r2_mean","mean"),
        mean_pearson=("pearson_mean","mean"),
        mean_best_discovered=("best_discovered_mean","mean"),
        mean_top20_discovered=("top20_mean_discovered_mean","mean"),
        mean_discovered=("mean_discovered_mean","mean"),
    )
    allr = ranks_all.groupby("model", as_index=False).agg(
        prediction_rank_all=("prediction_composite_rank","mean"),
        discovery_rank_all=("discovery_composite_rank","mean"),
        overall_rank_all=("overall_composite_rank","mean"),
    )
    uqr = ranks_uq.groupby("model", as_index=False).agg(
        prediction_rank_uncertainty_aware=("prediction_composite_rank","mean"),
        discovery_rank_uncertainty_aware=("discovery_composite_rank","mean"),
        overall_rank_uncertainty_aware=("overall_composite_rank","mean"),
    )
    return grouped.merge(allr, on="model", how="left").merge(uqr, on="model", how="left")

def round_rank_summary(round_summary, models):
    subset = round_summary[round_summary["model"].isin(models)].copy()
    needed = ["rmse_mean","top20_mean_discovered_mean"]
    if not all(c in subset.columns for c in needed):
        return pd.DataFrame()
    subset["rmse_rank"] = subset.groupby(["target","round"])["rmse_mean"].rank(
        method="average", ascending=True
    )
    subset["top20_rank"] = subset.groupby(["target","round"])["top20_mean_discovered_mean"].rank(
        method="average", ascending=False
    )
    subset["round_composite_rank"] = subset[["rmse_rank","top20_rank"]].mean(axis=1)
    return subset.groupby(["round","model"], as_index=False).agg(
        mean_rmse=("rmse_mean","mean"),
        std_rmse_across_targets=("rmse_mean","std"),
        mean_top20_discovered=("top20_mean_discovered_mean","mean"),
        std_top20_across_targets=("top20_mean_discovered_mean","std"),
        mean_rmse_rank=("rmse_rank","mean"),
        mean_top20_rank=("top20_rank","mean"),
        mean_composite_rank=("round_composite_rank","mean"),
    ).sort_values(["round","mean_composite_rank"])

def main():
    args = parse_args()
    root = args.input_root.resolve()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    config_problems = []
    for target in TARGETS:
        original, corrected = discover_target_runs(root, target)
        runs.extend([original, corrected])
        config_problems += validate_config(
            original, args.expected_beta, args.expected_seeds, args.expected_rounds
        )
        config_problems += validate_config(
            corrected, args.expected_beta, args.expected_seeds, args.expected_rounds
        )

    provenance_rows = []
    for run in runs:
        provenance_rows.append({
            "target": run.target,
            "source_run_type": run.run_type,
            "source_path": str(run.directory),
            "strategy": run.config.get("strategy"),
            "beta": run.config.get("beta"),
            "seeds": run.config.get("seeds"),
            "rounds": run.config.get("rounds"),
            "models": "|".join(str(x) for x in run.config.get("models", [])),
        })
    provenance = pd.DataFrame(provenance_rows)

    history = pd.concat([prepare_history(r) for r in runs], ignore_index=True, sort=False)
    round_summary = pd.concat(
        [prepare_summary(r, "round_summary.csv") for r in runs],
        ignore_index=True, sort=False
    )
    final = pd.concat(
        [prepare_summary(r, "final_round_summary.csv") for r in runs],
        ignore_index=True, sort=False
    )

    problems = (
        config_problems
        + validate_history(history, args.expected_seeds, args.expected_rounds)
        + validate_final(final)
    )

    report = out / "validation_report.txt"
    lines = [
        "MULTI-TARGET REGRESSION BENCHMARK VALIDATION",
        "="*50,
        f"Input root: {root}",
        f"Selected source directories: {len(runs)}",
        f"Canonical history rows: {len(history)}",
        f"Final target/model rows: {len(final)}",
        "",
    ]
    if problems:
        lines += ["FAILED", ""] + [f"- {p}" for p in problems]
    else:
        lines += ["PASSED", "", "All ten targets and eight models were found with no duplicate canonical history rows."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if problems:
        print(f"Validation failed. See {report}")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    provenance.to_csv(out / "provenance.csv", index=False)
    history.to_csv(out / "canonical_history.csv", index=False)
    round_summary.to_csv(out / "canonical_round_summary.csv", index=False)
    final.to_csv(out / "final_model_target_summary.csv", index=False)

    ranks_all = make_ranks(final, MODELS)
    ranks_uq = make_ranks(final, UNCERTAINTY_AWARE_MODELS)
    ranks_all.to_csv(out / "model_ranks_all.csv", index=False)
    ranks_uq.to_csv(out / "model_ranks_uncertainty_aware.csv", index=False)
    rank_summary(ranks_all).to_csv(out / "model_rank_summary_all.csv", index=False)
    rank_summary(ranks_uq).to_csv(out / "model_rank_summary_uncertainty_aware.csv", index=False)
    winners(final, MODELS).to_csv(out / "metric_winners_all.csv", index=False)
    winners(final, UNCERTAINTY_AWARE_MODELS).to_csv(
        out / "metric_winners_uncertainty_aware.csv", index=False
    )
    prediction_discovery(final, ranks_all, ranks_uq).to_csv(
        out / "prediction_discovery_summary.csv", index=False
    )
    round_rank_summary(round_summary, MODELS).to_csv(
        out / "round_rank_summary_all.csv", index=False
    )
    round_rank_summary(round_summary, UNCERTAINTY_AWARE_MODELS).to_csv(
        out / "round_rank_summary_uncertainty_aware.csv", index=False
    )

    print("Validation passed.")
    print(f"Canonical history rows: {len(history):,}")
    print(f"Final target/model rows: {len(final):,}")
    print(f"Outputs: {out}")

if __name__ == "__main__":
    main()
