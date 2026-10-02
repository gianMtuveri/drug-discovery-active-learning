#!/usr/bin/env python3
"""Diagnose pool-wide GP/UCB score degeneracy on the fixed v0.5.1 partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.regression.run_v051_representation_integration_smoke import _load_inputs
from src.regression.representation_surrogates import make_representation_surrogate
from src.regression.selection import (
    minmax_normalize,
    robust_scale,
    select_acquisition_topk,
)


DEFAULT_REPRESENTATIONS = ("unimol_v1",)
ALL_GP_REPRESENTATIONS = (
    "unimol_v1",
    "morgan",
    "rdkit_descriptors",
    "smi_ted_light",
)
OUTPUT_DIR = REPOSITORY_ROOT / "results/acquisition_degeneracy/v051"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=ALL_GP_REPRESENTATIONS,
        default=list(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _range_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return {
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_range": maximum - minimum,
        f"{prefix}_std": float(np.std(values)),
    }


def run_trajectory(
    representation_name: str,
    X_model: np.ndarray,
    y: np.ndarray,
    manifest,
    *,
    n_rounds: int,
) -> tuple[pd.DataFrame, dict]:
    pool_indices = np.asarray(manifest.acquisition_pool_indices, dtype=int)
    labelled = np.asarray(manifest.initial_labeled_indices, dtype=int)
    unlabelled = np.setdiff1d(pool_indices, labelled, assume_unique=False)
    records: list[dict] = []
    started = time.perf_counter()

    for round_idx in range(n_rounds):
        model = make_representation_surrogate(
            "gaussian_process", representation_name, random_state=round_idx
        )
        model.fit(X_model[labelled], y[labelled])
        prediction, uncertainty = model.predict_with_uncertainty(X_model[unlabelled])
        prediction_scaled = robust_scale(prediction)
        uncertainty_scaled = robust_scale(uncertainty)
        raw_ucb = prediction_scaled + 2.0 * uncertainty_scaled
        normalized_ucb = minmax_normalize(raw_ucb)

        unique_scores, counts = np.unique(normalized_ucb, return_counts=True)
        largest_tie = int(np.max(counts))
        complete_tie = len(unique_scores) == 1
        selected = select_acquisition_topk(unlabelled, normalized_ucb, batch_size=10)
        selected_positions = np.asarray(
            [np.flatnonzero(unlabelled == index)[0] for index in selected], dtype=int
        )
        score_counts = dict(zip(unique_scores.tolist(), counts.tolist()))
        selected_tied_count = sum(
            score_counts[float(normalized_ucb[position])] > 1
            for position in selected_positions
        )
        if complete_tie:
            selection_mode = "complete_tie_fallback"
        elif selected_tied_count:
            selection_mode = "partially_affected_by_ties"
        else:
            selection_mode = "model_ranked"

        record = {
            "representation": representation_name,
            "round": round_idx,
            "n_labelled": int(len(labelled)),
            "n_pool": int(len(unlabelled)),
            **_range_stats(prediction, "prediction"),
            **_range_stats(uncertainty, "uncertainty"),
            **_range_stats(raw_ucb, "raw_ucb"),
            **_range_stats(normalized_ucb, "normalized_ucb"),
            "normalized_ucb_unique_count": int(len(unique_scores)),
            "prediction_informative": bool(not np.all(prediction_scaled == 0.0)),
            "uncertainty_informative": bool(not np.all(uncertainty_scaled == 0.0)),
            "complete_tie": bool(complete_tie),
            "largest_tie_count": largest_tie,
            "tie_fraction": largest_tie / len(unlabelled),
            "selected_tied_count": int(selected_tied_count),
            "selection_mode": selection_mode,
            "selected_indices": json.dumps(selected.tolist()),
        }
        if not all(
            np.isfinite(record[key])
            for key in (
                "prediction_range", "prediction_std", "uncertainty_range",
                "uncertainty_std", "raw_ucb_range", "raw_ucb_std",
                "normalized_ucb_range", "normalized_ucb_std",
            )
        ):
            raise RuntimeError(f"Non-finite diagnostics in round {round_idx}")
        records.append(record)
        print(
            f"{representation_name}: round {round_idx:02d}/{n_rounds - 1:02d}, "
            f"pool={len(unlabelled)}, mode={selection_mode}",
            flush=True,
        )
        labelled = np.concatenate([labelled, selected])
        unlabelled = np.setdiff1d(unlabelled, selected, assume_unique=False)

    frame = pd.DataFrame(records)
    collapsed = frame.index[frame.complete_tie].tolist()
    first_collapsed = int(frame.loc[collapsed[0], "round"]) if collapsed else None
    recovered = False
    persistent = False
    if first_collapsed is not None:
        later = frame.loc[frame["round"] > first_collapsed, "complete_tie"]
        recovered = bool((~later).any())
        persistent = bool(len(later) > 0 and later.all())
    summary = {
        "representation": representation_name,
        "model": "gaussian_process",
        "acquisition": "ucb",
        "seed": 0,
        "rounds": n_rounds,
        "first_collapsed_round": first_collapsed,
        "collapsed_rounds": int(frame.complete_tie.sum()),
        "recovered_after_first_collapse": recovered,
        "persistent_after_first_collapse": persistent,
        "tie_fallback_acquisitions": int(
            (frame.selection_mode == "complete_tie_fallback").sum()
        ),
        "partially_tied_rounds": int(
            (frame.selection_mode == "partially_affected_by_ties").sum()
        ),
        "runtime_seconds": time.perf_counter() - started,
        "collapse_scope": "entire remaining acquisition pool",
    }
    return frame, summary


def main() -> None:
    args = parse_args()
    representations, mask, ids, manifest, y, _, _, _ = _load_inputs()
    frames = []
    summaries = []
    for representation_name in args.representations:
        X_model = np.asarray(representations[representation_name].features[mask])
        frame, summary = run_trajectory(
            representation_name, X_model, y, manifest, n_rounds=args.rounds
        )
        frames.append(frame)
        summaries.append(summary)
        print(
            f"{representation_name}: {summary['collapsed_rounds']}/{args.rounds} "
            f"complete-tie rounds in {summary['runtime_seconds']:.2f}s",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.concat(frames, ignore_index=True)
    diagnostics.to_parquet(args.output_dir / "per_round_diagnostics.parquet", index=False)
    diagnostics.to_csv(args.output_dir / "per_round_diagnostics.csv", index=False)
    payload = {
        "purpose": "acquisition degeneracy diagnostic; not a performance comparison",
        "common_population_size": len(ids),
        "common_population_hash": manifest.ordered_population_hash,
        "fixed_partition_seed": manifest.random_seed,
        "initial_labelled": manifest.n_initial,
        "batch_size": 10,
        "beta": 2.0,
        "tie_policy": "np.argsort(scores, kind='stable')[::-1]",
        "trajectories": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# v0.5.1 GP/UCB acquisition-score degeneracy diagnostic",
        "",
        "Diagnostic only; no predictive or discovery performance comparison was made.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['representation']}",
                "",
                f"- First collapsed round: {summary['first_collapsed_round']}",
                f"- Complete ties: {summary['collapsed_rounds']} / {args.rounds}",
                f"- Recovered after collapse: {summary['recovered_after_first_collapse']}",
                f"- Persistent after collapse: {summary['persistent_after_first_collapse']}",
                f"- Runtime: {summary['runtime_seconds']:.2f} seconds",
                "",
            ]
        )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
