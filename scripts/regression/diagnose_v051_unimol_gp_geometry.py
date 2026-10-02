#!/usr/bin/env python3
"""Limited geometry check for representative Uni-Mol GP trajectory rounds."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regression.run_v051_representation_integration_smoke import _load_inputs
from src.regression.representation_surrogates import make_representation_surrogate
from src.regression.selection import minmax_normalize, robust_scale, select_acquisition_topk


ROUNDS = frozenset((1, 2, 4, 6))
OUTPUT = ROOT / "results/acquisition_degeneracy/v051/gp_state/geometry.parquet"


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_q25": float(q25),
        f"{prefix}_median": float(median),
        f"{prefix}_q75": float(q75),
        f"{prefix}_max": float(values.max()),
    }


def main() -> None:
    representations, mask, _, manifest, y, _, _, _ = _load_inputs()
    X = np.asarray(representations["unimol_v1"].features[mask])
    labelled = np.asarray(manifest.initial_labeled_indices, dtype=int)
    unlabelled = np.setdiff1d(np.asarray(manifest.acquisition_pool_indices), labelled)
    records = []
    for round_idx in range(7):
        model = make_representation_surrogate(
            "gaussian_process", "unimol_v1", random_state=round_idx
        )
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            model.fit(X[labelled], y[labelled])
        prediction, uncertainty = model.predict_with_uncertainty(X[unlabelled])
        normalized = minmax_normalize(
            robust_scale(prediction) + 2.0 * robust_scale(uncertainty)
        )
        selected = select_acquisition_topk(unlabelled, normalized, 10)

        if round_idx in ROUNDS:
            transformed_labelled = model.preprocessor.transform(X[labelled])
            transformed_pool = model.preprocessor.transform(X[unlabelled])
            distances = pairwise_distances(transformed_pool, transformed_labelled)
            nearest = distances.min(axis=1)
            fitted = model.model.kernel_
            length_scale = float(np.asarray(fitted.k1.k2.length_scale).item())
            signal_variance = float(fitted.k1.k1.constant_value)
            maximum_rbf_similarity = np.exp(-0.5 * (nearest / length_scale) ** 2)
            maximum_signal_covariance = signal_variance * maximum_rbf_similarity
            records.append(
                {
                    "round": round_idx,
                    "n_labelled": len(labelled),
                    "retained_features": transformed_labelled.shape[1],
                    "length_scale": length_scale,
                    "signal_variance": signal_variance,
                    **distribution(nearest, "nearest_distance"),
                    **distribution(maximum_rbf_similarity, "max_rbf_similarity"),
                    **distribution(maximum_signal_covariance, "max_signal_covariance"),
                    "zero_max_similarity_fraction": float(
                        np.mean(maximum_rbf_similarity == 0.0)
                    ),
                }
            )
            print(
                f"round {round_idx}: distance median={np.median(nearest):.4g}; "
                f"max-similarity median={np.median(maximum_rbf_similarity):.4g}",
                flush=True,
            )
        labelled = np.concatenate([labelled, selected])
        unlabelled = np.setdiff1d(unlabelled, selected)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(OUTPUT, index=False)
    (OUTPUT.with_suffix(".json")).write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
