"""Reproducible, metadata-driven analysis of the v0.5.0 EGFR benchmark.

This module never infers experimental identity from paths or result values.
Identity is accepted only when ``config.json`` and the corresponding history
columns agree.  It operates entirely on saved benchmark artifacts.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


STRATEGIES = ("ucb", "ucb_qed", "ucb_diversity", "ucb_qed_diversity")
MODELS = ("random_forest", "extra_trees", "bayesian_ridge", "gaussian_process")
CAMPAIGN_KEY = ["strategy", "model", "seed"]
RECORD_KEY = CAMPAIGN_KEY + ["round"]
FINAL_METRICS = [
    "joint_discovery_count",
    "joint_discovery_auc",
    "affinity_discovery_count",
    "joint_scaffold_count",
    "joint_diversity",
    "rmse",
    "pearson",
    "qed_fraction_of_affinity_discoveries",
]


@dataclass(frozen=True)
class LoadResult:
    history: pd.DataFrame
    sources: pd.DataFrame
    overlaps: pd.DataFrame


def parse_list(value: object) -> list:
    """Parse a JSON/Python-list CSV cell without evaluating arbitrary code."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = ast.literal_eval(text)
    if not isinstance(result, list):
        raise ValueError(f"Expected a serialized list, got {type(result).__name__}")
    return result


def identify_source(directory: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load one artifact directory and prove config/history identity agreement."""
    directory = Path(directory)
    config_path = directory / "config.json"
    history_path = directory / "history.csv"
    if not config_path.is_file() or not history_path.is_file():
        raise ValueError(f"Artifact source lacks config.json or history.csv: {directory}")
    config = json.loads(config_path.read_text())
    missing_config = {"strategy", "models", "target"} - set(config)
    if missing_config:
        raise ValueError(f"{config_path} lacks identity metadata: {sorted(missing_config)}")
    strategy = config["strategy"]
    if strategy not in STRATEGIES:
        raise ValueError(f"Unexpected strategy in {config_path}: {strategy!r}")
    configured_models = set(config["models"])
    if not configured_models or not configured_models <= set(MODELS):
        raise ValueError(f"Unexpected model metadata in {config_path}: {config['models']!r}")
    history = pd.read_csv(history_path)
    required = set(RECORD_KEY + ["target"])
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"{history_path} lacks identity columns: {sorted(missing)}")
    if set(history["strategy"].dropna().unique()) != {strategy}:
        raise ValueError(f"Strategy conflict between {config_path} and {history_path}")
    row_models = set(history["model"].dropna().unique())
    if not row_models or not row_models <= configured_models:
        raise ValueError(f"Model conflict between {config_path} and {history_path}")
    if set(history["target"].dropna().str.upper().unique()) != {str(config["target"]).upper()}:
        raise ValueError(f"Target conflict between {config_path} and {history_path}")
    if history.duplicated(RECORD_KEY).any():
        raise ValueError(f"Duplicate campaign records inside {history_path}")
    history["source_directory"] = str(directory)
    history["source_config"] = str(config_path)
    return config, history


def _campaign_score(group: pd.DataFrame) -> tuple[int, int]:
    return int(group["round"].max()), int(group["round"].nunique())


def load_artifacts(root: str | Path) -> LoadResult:
    """Discover sources and retain the most complete source for each campaign.

    Overlapping sources are never concatenated blindly.  Molecular acquisition
    provenance must agree over overlapping rounds; otherwise loading stops.
    """
    root = Path(root)
    candidates = sorted(p.parent for p in root.glob("*/config.json"))
    if not candidates:
        raise ValueError(f"No benchmark artifact directories found below {root}")
    frames: list[pd.DataFrame] = []
    source_rows: list[dict] = []
    for directory in candidates:
        config, frame = identify_source(directory)
        frames.append(frame)
        source_rows.append({
            "source_directory": str(directory),
            "strategy": config["strategy"],
            "configured_models": json.dumps(config["models"]),
            "configured_seeds": config.get("n_seeds", config.get("seeds")),
            "configured_rounds": config.get("rounds", config.get("max_round")),
            "history_rows": len(frame),
            "history_models": json.dumps(sorted(frame["model"].unique())),
            "history_seeds": int(frame["seed"].nunique()),
            "history_max_round": int(frame["round"].max()),
        })
    raw = pd.concat(frames, ignore_index=True)
    overlap_rows: list[dict] = []
    retained: list[pd.DataFrame] = []
    provenance_columns = [
        c for c in ("initial_labeled_indices", "newly_acquired_indices",
                    "cumulative_acquired_indices") if c in raw.columns
    ]
    for key, campaign in raw.groupby(CAMPAIGN_KEY, sort=True):
        by_source = {s: g.copy() for s, g in campaign.groupby("source_directory")}
        ranked = sorted(by_source, key=lambda s: _campaign_score(by_source[s]), reverse=True)
        chosen = ranked[0]
        chosen_frame = by_source[chosen]
        for other in ranked[1:]:
            left = chosen_frame.merge(by_source[other], on=RECORD_KEY, suffixes=("_chosen", "_other"))
            agrees = True
            for column in provenance_columns:
                agrees &= bool((left[f"{column}_chosen"].fillna("").astype(str) ==
                                left[f"{column}_other"].fillna("").astype(str)).all())
            overlap_rows.append({
                **dict(zip(CAMPAIGN_KEY, key)),
                "chosen_source": chosen,
                "overlapping_source": other,
                "overlapping_records": len(left),
                "molecular_provenance_agrees": agrees,
            })
            if len(left) and provenance_columns and not agrees:
                raise ValueError(
                    "Conflicting molecular provenance in overlapping sources for "
                    f"strategy/model/seed={key}: {chosen} versus {other}"
                )
        retained.append(chosen_frame)
    history = pd.concat(retained, ignore_index=True).sort_values(RECORD_KEY).reset_index(drop=True)
    if history.duplicated(RECORD_KEY).any():
        raise AssertionError("Source resolution left duplicate campaign records")
    return LoadResult(history, pd.DataFrame(source_rows), pd.DataFrame(overlap_rows))


def add_recalculated_endpoints(history: pd.DataFrame) -> pd.DataFrame:
    """Recalculate endpoints possible from row-level saved molecule provenance."""
    out = history.copy()
    recalculated: list[dict] = []
    for _, row in out.iterrows():
        affinity = np.asarray(parse_list(row.get("newly_acquired_affinity_nm")), dtype=float)
        qed = np.asarray(parse_list(row.get("newly_acquired_qed")), dtype=float)
        smiles = [str(x) for x in parse_list(row.get("newly_acquired_smiles"))]
        if not (len(affinity) == len(qed) == len(smiles)):
            raise ValueError(f"Misaligned acquired molecule values for {tuple(row[k] for k in RECORD_KEY)}")
        active = affinity <= 500.0
        joint = active & (qed >= 0.6)
        scaffold_set: set[str] = set()
        fps = []
        for smi in np.asarray(smiles)[joint]:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise ValueError(f"Invalid acquired SMILES: {smi}")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold:
                scaffold_set.add(scaffold)
            fps.append(AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol))
        batch_diversity = np.nan
        if len(fps) >= 2:
            similarities = []
            for i in range(1, len(fps)):
                similarities.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
            batch_diversity = float(1.0 - np.mean(similarities))
        recalculated.append({
            "batch_affinity_discovery_count_recalculated": int(active.sum()),
            "batch_joint_discovery_count_recalculated": int(joint.sum()),
            "batch_joint_scaffold_count_recalculated": len(scaffold_set),
            "batch_joint_diversity_morgan_r2_2048": batch_diversity,
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(recalculated)], axis=1)


def campaign_endpoints(history: pd.DataFrame, expected_rounds: int = 20) -> pd.DataFrame:
    """Return one final-row record per campaign, with derived QED fraction."""
    rows = []
    for key, group in history.groupby(CAMPAIGN_KEY, sort=True):
        group = group.sort_values("round")
        final = group.iloc[-1].copy()
        final["observed_rounds"] = int(group["round"].max())
        final["n_round_rows"] = int(group["round"].nunique())
        final["complete_campaign"] = (
            set(group["round"].astype(int)) == set(range(expected_rounds + 1))
        )
        affinity = float(final["affinity_discovery_count"])
        final["qed_fraction_of_affinity_discoveries"] = (
            float(final["joint_discovery_count"]) / affinity if affinity else np.nan
        )
        rows.append(final)
    return pd.DataFrame(rows).reset_index(drop=True)


def validate(history: pd.DataFrame, overlaps: pd.DataFrame) -> pd.DataFrame:
    """Produce explicit passed/failed/unverifiable validation results."""
    checks: list[dict] = []
    def add(check: str, status: str, details: str) -> None:
        checks.append({"check": check, "status": status, "details": details})

    actual_strategies = set(history["strategy"].unique())
    actual_models = set(history["model"].unique())
    add("strategy identities", "passed" if actual_strategies == set(STRATEGIES) else "failed",
        f"observed={sorted(actual_strategies)}; identities verified against config and history")
    add("model identities", "passed" if actual_models == set(MODELS) else "failed",
        f"observed={sorted(actual_models)}; identities verified against config and history")
    endpoints = campaign_endpoints(history)
    expected_campaigns = {(s, m, seed) for s in STRATEGIES for m in MODELS for seed in range(20)}
    actual_campaigns = set(map(tuple, endpoints[CAMPAIGN_KEY].itertuples(index=False, name=None)))
    missing = expected_campaigns - actual_campaigns
    add("twenty seeds per strategy/model", "passed" if not missing and len(actual_campaigns) == 320 else "failed",
        f"observed {len(actual_campaigns)}/320 campaigns; missing={len(missing)}")
    incomplete = endpoints.loc[~endpoints["complete_campaign"], CAMPAIGN_KEY + ["observed_rounds"]]
    add("twenty acquisition rounds plus initialization", "passed" if incomplete.empty else "failed",
        "all campaigns contain rounds 0..20" if incomplete.empty else
        f"{len(incomplete)} incomplete campaigns; all are listed in incomplete_campaigns.csv")
    initial_ok = bool((history["n_initial_labeled"] == 20).all())
    batches = history.loc[history["round"] > 0, "n_newly_acquired"]
    batch_ok = bool((batches == 10).all())
    add("initial and batch sizes", "passed" if initial_ok and batch_ok else "failed",
        f"n_initial=20 throughout: {initial_ok}; n_newly_acquired=10 after round 0: {batch_ok}")
    duplicate_ok = not history.duplicated(RECORD_KEY).any()
    add("no duplicate combined campaign records", "passed" if duplicate_ok else "failed",
        f"retained duplicates={int(history.duplicated(RECORD_KEY).sum())}; source overlaps={len(overlaps)}")
    initial_sets = history.loc[history["round"] == 0].groupby("seed")["initial_labeled_indices"].nunique()
    init_match = bool((initial_sets == 1).all()) and set(initial_sets.index) == set(range(20))
    add("identical initial labelled sets within seed", "passed" if init_match else "failed",
        f"stored molecule-index provenance agrees: {bool((initial_sets == 1).all())}; seeds observed={len(initial_sets)}")
    add("identical acquisition pools within seed", "unverifiable",
        "pool membership was not serialized in CSV/config artifacts")
    add("identical held-out test partitions within seed", "unverifiable",
        "held-out molecule identities were not serialized in CSV/config artifacts")
    add("no held-out test molecules acquired", "unverifiable",
        "acquired identities are stored, but held-out test identities are absent")
    expected_labeled = 20 + 10 * history["round"]
    accounting = ((history["n_labeled"] == expected_labeled) &
                  (history["n_cumulative_acquired"] == 10 * history["round"]))
    add("measurement accounting", "passed" if accounting.all() else "failed",
        f"valid records={int(accounting.sum())}/{len(accounting)}")
    return pd.DataFrame(checks)


def validate_endpoint_consistency(history: pd.DataFrame) -> pd.DataFrame:
    """Check stored cumulative endpoints from saved per-batch molecule values."""
    rows = []
    for key, group in history.groupby(CAMPAIGN_KEY, sort=True):
        group = group.sort_values("round")
        cumulative_joint = 0
        cumulative_affinity = 0
        curve_x = []
        curve_y = []
        ok_counts = True
        ok_auc = True
        for _, row in group.iterrows():
            cumulative_joint += int(row["batch_joint_discovery_count_recalculated"])
            cumulative_affinity += int(row["batch_affinity_discovery_count_recalculated"])
            ok_counts &= cumulative_joint == int(row["joint_discovery_count"])
            ok_counts &= cumulative_affinity == int(row["affinity_discovery_count"])
            curve_x.append(float(row["n_cumulative_acquired"]))
            curve_y.append(float(cumulative_joint))
            auc = float(np.trapezoid(curve_y, curve_x)) if len(curve_x) > 1 else 0.0
            ok_auc &= np.isclose(auc, float(row["joint_discovery_auc"]), rtol=0, atol=1e-9)
        final = group.iloc[-1]
        rows.append({
            **dict(zip(CAMPAIGN_KEY, key)),
            "cumulative_counts_agree": ok_counts,
            "auc_agrees": ok_auc,
            "final_count_agrees_with_curve": cumulative_joint == int(final["joint_discovery_count"]),
        })
    return pd.DataFrame(rows)


def summarize_complete(endpoints: pd.DataFrame) -> pd.DataFrame:
    complete = endpoints.loc[endpoints["complete_campaign"]].copy()
    rows = []
    for (strategy, model), group in complete.groupby(["strategy", "model"], sort=True):
        row = {"strategy": strategy, "model": model, "n_seeds": group["seed"].nunique()}
        for metric in FINAL_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
            row[f"{metric}_min"] = group[metric].min()
            row[f"{metric}_max"] = group[metric].max()
        rows.append(row)
    return pd.DataFrame(rows)


COMPARISONS = {
    "adding_qed_without_diversity": ("ucb_qed", "ucb"),
    "adding_qed_with_diversity": ("ucb_qed_diversity", "ucb_diversity"),
    "adding_diversity_without_qed": ("ucb_diversity", "ucb"),
    "adding_diversity_with_qed": ("ucb_qed_diversity", "ucb_qed"),
}


def paired_comparisons(endpoints: pd.DataFrame,
                       metrics: Iterable[str] = FINAL_METRICS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate seed-matched strategy effects and QED×diversity interactions."""
    complete = endpoints.loc[endpoints["complete_campaign"]].copy()
    records = []
    for model, model_frame in complete.groupby("model", sort=True):
        indexed = model_frame.set_index(["seed", "strategy"])
        seeds = sorted(set(model_frame["seed"]))
        for seed in seeds:
            for name, (treatment, baseline) in COMPARISONS.items():
                if (seed, treatment) not in indexed.index or (seed, baseline) not in indexed.index:
                    continue
                for metric in metrics:
                    records.append({"model": model, "seed": seed, "comparison": name,
                                    "metric": metric,
                                    "treatment_strategy": treatment,
                                    "baseline_strategy": baseline,
                                    "treatment_value": indexed.loc[(seed, treatment), metric],
                                    "baseline_value": indexed.loc[(seed, baseline), metric],
                                    "paired_difference": indexed.loc[(seed, treatment), metric] - indexed.loc[(seed, baseline), metric]})
            needed = [(seed, s) for s in STRATEGIES]
            if all(k in indexed.index for k in needed):
                for metric in metrics:
                    interaction = (indexed.loc[(seed, "ucb_qed_diversity"), metric]
                                   - indexed.loc[(seed, "ucb_diversity"), metric]
                                   - indexed.loc[(seed, "ucb_qed"), metric]
                                   + indexed.loc[(seed, "ucb"), metric])
                    records.append({"model": model, "seed": seed,
                                    "comparison": "qed_by_diversity_interaction",
                                    "metric": metric, "treatment_strategy": "difference_in_differences",
                                    "baseline_strategy": "factorial_ucb_baseline",
                                    "treatment_value": np.nan, "baseline_value": np.nan,
                                    "paired_difference": interaction})
    seed_level = pd.DataFrame(records)
    summary = (seed_level.groupby(["model", "comparison", "metric"], sort=True)
               .agg(n_seeds=("seed", "nunique"),
                    mean_paired_difference=("paired_difference", "mean"),
                    sd_paired_difference=("paired_difference", "std"),
                    min_paired_difference=("paired_difference", "min"),
                    max_paired_difference=("paired_difference", "max"))
               .reset_index())
    return seed_level, summary
