#!/usr/bin/env python3
"""Controlled Uni-Mol GP kernel acquisition-degeneracy diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RationalQuadratic, WhiteKernel


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regression.run_v051_representation_integration_smoke import _load_inputs
from src.data.partitions import load_partition_manifest
from src.regression.representation_surrogates import make_representation_surrogate
from src.regression.selection import minmax_normalize, robust_scale, select_acquisition_topk


COMMON_DIR = ROOT / "data/processed/v051/common_population"
OUTPUT_DIR = ROOT / "results/acquisition_degeneracy/v051/gp_kernel_comparison"
BASELINE_PATH = ROOT / "results/acquisition_degeneracy/v051/gp_seed_sensitivity/per_round_seed_diagnostics.parquet"
SEED0_PATH = ROOT / "results/acquisition_degeneracy/v051/gp_state/per_round_gp_state.parquet"
BOUNDS = (1e-5, 1e5)
KERNELS = {
    "matern_1_5": lambda: ConstantKernel(1.0, BOUNDS) * Matern(
        length_scale=1.0, length_scale_bounds=BOUNDS, nu=1.5
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=BOUNDS),
    "matern_2_5": lambda: ConstantKernel(1.0, BOUNDS) * Matern(
        length_scale=1.0, length_scale_bounds=BOUNDS, nu=2.5
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=BOUNDS),
    "rational_quadratic": lambda: ConstantKernel(1.0, BOUNDS) * RationalQuadratic(
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=BOUNDS,
        alpha_bounds=BOUNDS,
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=BOUNDS),
}


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    return {
        f"{prefix}_min": low,
        f"{prefix}_max": high,
        f"{prefix}_range": high - low,
        f"{prefix}_std": float(values.std()),
    }


def _parameter_bounds(fitted) -> list[str]:
    states = []
    parameters = fitted.get_params(deep=True)
    for hyperparameter in fitted.hyperparameters:
        value = np.asarray(parameters[hyperparameter.name], dtype=float).reshape(-1)
        bounds = np.asarray(hyperparameter.bounds, dtype=float)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], len(value), axis=0)
        for position, (item, (lower, upper)) in enumerate(zip(value, bounds)):
            suffix = f"[{position}]" if len(value) > 1 else ""
            if np.isclose(item, lower, rtol=1e-5, atol=0.0):
                states.append(f"{hyperparameter.name}{suffix}:lower")
            elif np.isclose(item, upper, rtol=1e-5, atol=0.0):
                states.append(f"{hyperparameter.name}{suffix}:upper")
    return states


def validate_baseline() -> pd.DataFrame:
    baseline = pd.read_parquet(BASELINE_PATH).copy()
    if len(baseline) != 400 or baseline.groupby("seed").size().ne(20).any():
        raise RuntimeError("Existing RBF seed diagnostic is not 20 seeds x 20 rounds")
    reference = pd.read_parquet(SEED0_PATH).sort_values("round")
    seed_zero = baseline.loc[baseline.seed == 0].sort_values("round")
    for column in (
        "constant_value", "length_scale", "noise_level", "prediction_range",
        "uncertainty_range", "raw_ucb_range", "largest_tie_fraction",
    ):
        if not np.allclose(seed_zero[column], reference[column], rtol=1e-10, atol=1e-12):
            raise RuntimeError(f"Existing RBF baseline differs from seed-0 reference: {column}")
    for column in (
        "normalized_ucb_unique_count", "selected_tied_count",
        "selected_from_largest_tie_group", "complete_tie", "selected_indices",
    ):
        if seed_zero[column].tolist() != reference[column].tolist():
            raise RuntimeError(f"Existing RBF baseline differs from seed-0 reference: {column}")
    baseline["kernel_family"] = "rbf"
    baseline["rational_quadratic_alpha"] = np.nan
    return baseline


def run_seed(kernel_name: str, seed: int, X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
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
        model.estimator.model.kernel = KERNELS[kernel_name]()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X[labelled], y[labelled])
        prediction, uncertainty = model.predict_with_uncertainty(X[unlabelled])
        prediction_scaled = robust_scale(prediction)
        uncertainty_scaled = robust_scale(uncertainty)
        raw_ucb = prediction_scaled + 2.0 * uncertainty_scaled
        normalized = minmax_normalize(raw_ucb)
        if not all(np.isfinite(x).all() for x in (prediction, uncertainty, raw_ucb, normalized)):
            raise RuntimeError(f"Non-finite output for {kernel_name}, seed {seed}, round {round_idx}")
        unique, counts = np.unique(normalized, return_counts=True)
        selected = select_acquisition_topk(unlabelled, normalized, 10)
        positions = np.asarray(
            [np.flatnonzero(unlabelled == index)[0] for index in selected], dtype=int
        )
        frequencies = dict(zip(unique.tolist(), counts.tolist()))
        selected_tied = sum(
            frequencies[float(normalized[position])] > 1 for position in positions
        )
        largest_count = int(counts.max())
        largest_values = (
            {float(value) for value, count in zip(unique, counts) if count == largest_count}
            if largest_count > 1 else set()
        )
        selected_largest = sum(
            float(normalized[position]) in largest_values for position in positions
        )
        fitted = model.model.kernel_
        rq_alpha = (
            float(fitted.k1.k2.alpha) if kernel_name == "rational_quadratic" else np.nan
        )
        warning_records = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
        records.append(
            {
                "kernel_family": kernel_name,
                "seed": seed,
                "round": round_idx,
                "n_labelled": int(len(labelled)),
                "n_pool": int(len(unlabelled)),
                "retained_features": int(model.preprocessor.retained_feature_mask_.sum()),
                "fitted_kernel": str(fitted),
                "constant_value": float(fitted.k1.k1.constant_value),
                "length_scale": float(np.asarray(fitted.k1.k2.length_scale).item()),
                "noise_level": float(fitted.k2.noise_level),
                "rational_quadratic_alpha": rq_alpha,
                "log_marginal_likelihood": float(model.model.log_marginal_likelihood_value_),
                "optimizer_warning": bool(warning_records),
                "optimizer_warnings": json.dumps(warning_records),
                "parameters_at_bounds": json.dumps(_parameter_bounds(fitted)),
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
            raise RuntimeError(f"Invalid acquisition for {kernel_name}, seed {seed}, round {round_idx}")
        acquired.extend(int(index) for index in selected)
        if len(acquired) != len(set(acquired)):
            raise RuntimeError(f"Repeated acquisition for {kernel_name}, seed {seed}")
        labelled = np.concatenate([labelled, selected])
        unlabelled = np.setdiff1d(unlabelled, selected)
    return pd.DataFrame(records)


def main() -> None:
    baseline = validate_baseline()
    print("RBF baseline: exact seed-0 reference validation passed", flush=True)
    representations, mask, ids, _, y, _, _, _ = _load_inputs()
    X = np.asarray(representations["unimol_v1"].features[mask])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alternative_path = OUTPUT_DIR / "alternative_kernel_rounds.parquet"
    alternative = pd.DataFrame()
    if alternative_path.exists():
        alternative = pd.read_parquet(alternative_path)
    started = time.perf_counter()
    for kernel_name in KERNELS:
        for seed in range(20):
            existing = alternative.loc[
                (alternative.kernel_family == kernel_name) & (alternative.seed == seed)
            ] if len(alternative) else pd.DataFrame()
            if len(existing) == 20:
                print(f"{kernel_name} seed {seed:02d}: checkpoint already complete", flush=True)
                continue
            seed_started = time.perf_counter()
            result = run_seed(kernel_name, seed, X, y, ids)
            if len(alternative):
                alternative = alternative.loc[
                    ~((alternative.kernel_family == kernel_name) & (alternative.seed == seed))
                ]
            alternative = pd.concat([alternative, result], ignore_index=True)
            alternative.to_parquet(alternative_path, index=False)
            print(
                f"{kernel_name} seed {seed:02d}: complete ties={int(result.complete_tie.sum())}, "
                f"tie rounds={int((result.selected_tied_count > 0).sum())}, "
                f"runtime={time.perf_counter() - seed_started:.2f}s",
                flush=True,
            )
    combined = pd.concat([baseline, alternative], ignore_index=True, sort=False)
    combined.to_parquet(OUTPUT_DIR / "per_round_kernel_diagnostics.parquet", index=False)
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "diagnostic-only controlled GP kernel comparison",
                "rbf_baseline_reused_and_validated": True,
                "kernel_initialization": {
                    "constant_value": 1.0,
                    "length_scale": 1.0,
                    "white_noise": 1.0,
                    "rational_quadratic_alpha": 1.0,
                },
                "common_bounds": list(BOUNDS),
                "total_runtime_seconds_alternatives": time.perf_counter() - started,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
