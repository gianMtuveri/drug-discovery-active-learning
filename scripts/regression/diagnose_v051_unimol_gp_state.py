#!/usr/bin/env python3
"""Inspect fitted GP state along the frozen Uni-Mol seed-0 trajectory."""

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
from src.regression.representation_surrogates import make_representation_surrogate
from src.regression.selection import minmax_normalize, robust_scale, select_acquisition_topk


OUTPUT_DIR = ROOT / "results/acquisition_degeneracy/v051/gp_state"
REFERENCE_PATH = ROOT / "results/acquisition_degeneracy/v051/per_round_diagnostics.parquet"


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    return {
        f"{prefix}_min": low,
        f"{prefix}_max": high,
        f"{prefix}_range": high - low,
        f"{prefix}_std": float(values.std()),
    }


def boundary_state(name: str, value: float, bounds: tuple[float, float]) -> str | None:
    lower, upper = bounds
    if np.isclose(value, lower, rtol=1e-5, atol=0.0):
        return f"{name}:lower"
    if np.isclose(value, upper, rtol=1e-5, atol=0.0):
        return f"{name}:upper"
    return None


def main() -> None:
    representations, mask, _, manifest, y, _, _, _ = _load_inputs()
    X = np.asarray(representations["unimol_v1"].features[mask])
    labelled = np.asarray(manifest.initial_labeled_indices, dtype=int)
    unlabelled = np.setdiff1d(
        np.asarray(manifest.acquisition_pool_indices, dtype=int), labelled
    )
    reference = pd.read_parquet(REFERENCE_PATH)
    reference = reference.loc[reference.representation == "unimol_v1"].set_index("round")
    records = []
    started = time.perf_counter()

    for round_idx in range(20):
        model = make_representation_surrogate(
            "gaussian_process", "unimol_v1", random_state=round_idx
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X[labelled], y[labelled])

        prediction, uncertainty = model.predict_with_uncertainty(X[unlabelled])
        prediction_scaled = robust_scale(prediction)
        uncertainty_scaled = robust_scale(uncertainty)
        raw_ucb = prediction_scaled + 2.0 * uncertainty_scaled
        normalized = minmax_normalize(raw_ucb)
        unique, counts = np.unique(normalized, return_counts=True)
        selected = select_acquisition_topk(unlabelled, normalized, batch_size=10)
        selected_positions = np.asarray(
            [np.flatnonzero(unlabelled == index)[0] for index in selected], dtype=int
        )
        frequency = dict(zip(unique.tolist(), counts.tolist()))
        selected_tied = sum(
            frequency[float(normalized[position])] > 1 for position in selected_positions
        )
        largest_count = int(counts.max())
        largest_values = (
            {
                float(value)
                for value, count in zip(unique, counts)
                if count == largest_count
            }
            if largest_count > 1
            else set()
        )
        selected_from_largest = sum(
            float(normalized[position]) in largest_values for position in selected_positions
        )

        expected = json.loads(reference.loc[round_idx, "selected_indices"])
        if selected.tolist() != expected:
            raise RuntimeError(
                f"Round {round_idx} selection differs from the saved diagnostic trajectory"
            )

        fitted = model.model.kernel_
        constant = float(fitted.k1.k1.constant_value)
        length_scale = float(np.asarray(fitted.k1.k2.length_scale).item())
        noise = float(fitted.k2.noise_level)
        bounds = {
            "constant": tuple(fitted.k1.k1.constant_value_bounds),
            "length_scale": tuple(fitted.k1.k2.length_scale_bounds),
            "noise": tuple(fitted.k2.noise_level_bounds),
        }
        at_bounds = [
            state for state in (
                boundary_state("constant", constant, bounds["constant"]),
                boundary_state("length_scale", length_scale, bounds["length_scale"]),
                boundary_state("noise", noise, bounds["noise"]),
            ) if state is not None
        ]
        warning_records = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
        target = y[labelled]
        record = {
            "round": round_idx,
            "n_labelled": int(len(labelled)),
            "n_pool": int(len(unlabelled)),
            "retained_features": int(model.preprocessor.retained_feature_mask_.sum()),
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
            "fitted_kernel": str(fitted),
            "constant_value": constant,
            "length_scale": length_scale,
            "noise_level": noise,
            "parameters_at_bounds": json.dumps(at_bounds),
            "optimizer_warning": bool(warning_records),
            "optimizer_warnings": json.dumps(warning_records),
            "log_marginal_likelihood": float(model.model.log_marginal_likelihood_value_),
            **stats(prediction, "prediction"),
            **stats(uncertainty, "uncertainty"),
            **stats(raw_ucb, "raw_ucb"),
            "normalized_ucb_unique_count": int(len(unique)),
            "largest_tie_fraction": largest_count / len(unlabelled),
            "selected_tied_count": int(selected_tied),
            "selected_from_largest_tie_group": int(selected_from_largest),
            "complete_tie": bool(len(unique) == 1),
            "selected_indices": json.dumps(selected.tolist()),
        }
        records.append(record)
        print(
            f"round {round_idx:02d}: kernel={fitted}; unique={len(unique)}; "
            f"warnings={len(warning_records)}",
            flush=True,
        )
        labelled = np.concatenate([labelled, selected])
        unlabelled = np.setdiff1d(unlabelled, selected)

    frame = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_DIR / "per_round_gp_state.parquet", index=False)
    frame.to_csv(OUTPUT_DIR / "per_round_gp_state.csv", index=False)
    summary = {
        "purpose": "diagnostic only; fitted GP mechanism",
        "trajectory_matches_saved_diagnostic": True,
        "runtime_seconds": time.perf_counter() - started,
        "rounds": 20,
        "warning_rounds": frame.loc[frame.optimizer_warning, "round"].tolist(),
        "bound_state_counts": frame.parameters_at_bounds.value_counts().to_dict(),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
