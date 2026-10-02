#!/usr/bin/env python3
"""Diagnose Uni-Mol GP/UCB posterior degeneracy across frozen seeds 0--19."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regression.run_v051_representation_integration_smoke import _load_inputs
from src.data.partitions import load_partition_manifest
from src.regression.representation_surrogates import make_representation_surrogate
from src.regression.selection import minmax_normalize, robust_scale, select_acquisition_topk


COMMON_DIR = ROOT / "data/processed/v051/common_population"
OUTPUT_DIR = ROOT / "results/acquisition_degeneracy/v051/gp_seed_sensitivity"
SEED0_REFERENCE = ROOT / "results/acquisition_degeneracy/v051/gp_state/per_round_gp_state.parquet"


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    return {
        f"{prefix}_min": low,
        f"{prefix}_max": high,
        f"{prefix}_range": high - low,
        f"{prefix}_std": float(values.std()),
    }


def _bounds(fitted) -> list[str]:
    states = []
    entries = (
        ("constant", float(fitted.k1.k1.constant_value), fitted.k1.k1.constant_value_bounds),
        ("length_scale", float(np.asarray(fitted.k1.k2.length_scale).item()), fitted.k1.k2.length_scale_bounds),
        ("noise", float(fitted.k2.noise_level), fitted.k2.noise_level_bounds),
    )
    for name, value, (lower, upper) in entries:
        if np.isclose(value, lower, rtol=1e-5, atol=0.0):
            states.append(f"{name}:lower")
        elif np.isclose(value, upper, rtol=1e-5, atol=0.0):
            states.append(f"{name}:upper")
    return states


def _validate_seed_zero(frame: pd.DataFrame) -> None:
    reference = pd.read_parquet(SEED0_REFERENCE).sort_values("round")
    current = frame.sort_values("round")
    if not (len(reference) == len(current) == 20):
        raise RuntimeError("Seed-0 reference or reproduction does not contain 20 rounds")
    numeric = (
        "constant_value", "length_scale", "noise_level", "prediction_range",
        "uncertainty_range", "raw_ucb_range", "largest_tie_fraction",
    )
    for column in numeric:
        if not np.allclose(
            current[column], reference[column], rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(f"Seed 0 differs from reference for {column}")
    exact = (
        "normalized_ucb_unique_count", "selected_tied_count",
        "selected_from_largest_tie_group", "complete_tie", "selected_indices",
    )
    for column in exact:
        if current[column].tolist() != reference[column].tolist():
            raise RuntimeError(f"Seed 0 differs from reference for {column}")
    if not bool(current.loc[current["round"] == 1, "complete_tie"].iloc[0]):
        raise RuntimeError("Seed 0 round 1 did not reproduce the complete collapse")


def run_seed(seed: int, X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    manifest = load_partition_manifest(
        COMMON_DIR / "partitions" / f"seed_{seed:02d}.json", ids
    )
    pool = np.asarray(manifest.acquisition_pool_indices, dtype=int)
    test = set(manifest.held_out_test_indices)
    labelled = np.asarray(manifest.initial_labeled_indices, dtype=int)
    unlabelled = np.setdiff1d(pool, labelled)
    acquired: list[int] = []
    records = []

    for round_idx in range(20):
        model = make_representation_surrogate(
            "gaussian_process", "unimol_v1", random_state=seed + round_idx
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X[labelled], y[labelled])
        prediction, uncertainty = model.predict_with_uncertainty(X[unlabelled])
        prediction_scaled = robust_scale(prediction)
        uncertainty_scaled = robust_scale(uncertainty)
        raw_ucb = prediction_scaled + 2.0 * uncertainty_scaled
        normalized = minmax_normalize(raw_ucb)
        if not all(
            np.isfinite(values).all()
            for values in (prediction, uncertainty, raw_ucb, normalized)
        ):
            raise RuntimeError(f"Non-finite model output for seed {seed}, round {round_idx}")

        unique, counts = np.unique(normalized, return_counts=True)
        selected = select_acquisition_topk(unlabelled, normalized, batch_size=10)
        positions = np.asarray(
            [np.flatnonzero(unlabelled == index)[0] for index in selected], dtype=int
        )
        frequency = dict(zip(unique.tolist(), counts.tolist()))
        selected_tied = sum(
            frequency[float(normalized[position])] > 1 for position in positions
        )
        largest_count = int(counts.max())
        largest_values = (
            {
                float(value) for value, count in zip(unique, counts)
                if count == largest_count
            }
            if largest_count > 1 else set()
        )
        selected_largest = sum(
            float(normalized[position]) in largest_values for position in positions
        )
        fitted = model.model.kernel_
        warning_records = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
        records.append(
            {
                "seed": seed,
                "round": round_idx,
                "n_labelled": int(len(labelled)),
                "n_pool": int(len(unlabelled)),
                "retained_features": int(model.preprocessor.retained_feature_mask_.sum()),
                "constant_value": float(fitted.k1.k1.constant_value),
                "length_scale": float(np.asarray(fitted.k1.k2.length_scale).item()),
                "noise_level": float(fitted.k2.noise_level),
                "log_marginal_likelihood": float(model.model.log_marginal_likelihood_value_),
                "optimizer_warning": bool(warning_records),
                "optimizer_warnings": json.dumps(warning_records),
                "parameters_at_bounds": json.dumps(_bounds(fitted)),
                **_stats(prediction, "prediction"),
                **_stats(uncertainty, "uncertainty"),
                **_stats(raw_ucb, "raw_ucb"),
                "normalized_ucb_unique_count": int(len(unique)),
                "largest_tie_count": largest_count,
                "largest_tie_fraction": largest_count / len(unlabelled),
                "selected_tied_count": int(selected_tied),
                "selected_from_largest_tie_group": int(selected_largest),
                "complete_tie": bool(len(unique) == 1),
                "selected_indices": json.dumps(selected.tolist()),
            }
        )
        if len(selected) != 10 or any(int(index) in test for index in selected):
            raise RuntimeError(f"Invalid acquisition for seed {seed}, round {round_idx}")
        acquired.extend(int(index) for index in selected)
        if len(acquired) != len(set(acquired)):
            raise RuntimeError(f"Repeated acquisition for seed {seed}, round {round_idx}")
        labelled = np.concatenate([labelled, selected])
        unlabelled = np.setdiff1d(unlabelled, selected)

    return pd.DataFrame(records)


def main() -> None:
    representations, mask, ids, _, y, _, _, _ = _load_inputs()
    X = np.asarray(representations["unimol_v1"].features[mask])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    started = time.perf_counter()
    for seed in range(20):
        seed_started = time.perf_counter()
        frame = run_seed(seed, X, y, ids)
        if seed == 0:
            _validate_seed_zero(frame)
            print("seed 00: exact reference validation passed", flush=True)
        frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "per_round_seed_diagnostics.parquet", index=False)
        print(
            f"seed {seed:02d}: complete ties={int(frame.complete_tie.sum())}, "
            f"tie-affected rounds={int((frame.selected_tied_count > 0).sum())}, "
            f"runtime={time.perf_counter() - seed_started:.2f}s",
            flush=True,
        )
    summary = {
        "purpose": "diagnostic only; Uni-Mol GP/UCB seed sensitivity",
        "seed_zero_reference_validation": True,
        "seeds": list(range(20)),
        "rounds_per_seed": 20,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
