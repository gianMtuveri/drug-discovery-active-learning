#!/usr/bin/env python3
"""Verify production GP against the frozen Rational Quadratic diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regression.run_v051_representation_integration_smoke import _load_inputs
from src.regression.representation_surrogates import make_representation_surrogate
from src.regression.selection import minmax_normalize, robust_scale, select_acquisition_topk


DIAGNOSTIC = ROOT / (
    "results/acquisition_degeneracy/v051/gp_kernel_comparison/"
    "per_round_kernel_diagnostics.parquet"
)


def main() -> None:
    representations, mask, _, manifest, y, _, _, _ = _load_inputs()
    X = np.asarray(representations["unimol_v1"].features[mask])
    reference = pd.read_parquet(DIAGNOSTIC)
    reference = reference.loc[
        (reference.kernel_family == "rational_quadratic") & (reference.seed == 0)
    ].set_index("round")
    labelled = np.asarray(manifest.initial_labeled_indices, dtype=int)
    unlabelled = np.setdiff1d(
        np.asarray(manifest.acquisition_pool_indices, dtype=int), labelled
    )

    for round_idx in range(2):
        model = make_representation_surrogate(
            "gaussian_process", "unimol_v1", random_state=round_idx
        )
        model.fit(X[labelled], y[labelled])
        prediction, uncertainty = model.predict_with_uncertainty(X[unlabelled])
        raw_ucb = robust_scale(prediction) + 2.0 * robust_scale(uncertainty)
        normalized = minmax_normalize(raw_ucb)
        selected = select_acquisition_topk(unlabelled, normalized, 10)
        fitted = model.model.kernel_
        expected = reference.loc[round_idx]
        observed = {
            "constant_value": float(fitted.k1.k1.constant_value),
            "length_scale": float(fitted.k1.k2.length_scale),
            "noise_level": float(fitted.k2.noise_level),
            "rational_quadratic_alpha": float(fitted.k1.k2.alpha),
            "prediction_range": float(prediction.max() - prediction.min()),
            "uncertainty_range": float(uncertainty.max() - uncertainty.min()),
            "raw_ucb_range": float(raw_ucb.max() - raw_ucb.min()),
        }
        for name, value in observed.items():
            if not np.isclose(value, expected[name], rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    f"Round {round_idx} differs from diagnostic for {name}: "
                    f"{value} != {expected[name]}"
                )
        if selected.tolist() != json.loads(expected.selected_indices):
            raise RuntimeError(f"Round {round_idx} selected indices differ")
        if len(np.unique(normalized)) != int(expected.normalized_ucb_unique_count):
            raise RuntimeError(f"Round {round_idx} score uniqueness differs")
        print(
            f"round {round_idx}: production matches diagnostic; "
            f"kernel={fitted}",
            flush=True,
        )
        labelled = np.concatenate([labelled, selected])
        unlabelled = np.setdiff1d(unlabelled, selected)


if __name__ == "__main__":
    main()
