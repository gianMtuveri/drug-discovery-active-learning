#!/usr/bin/env python3
"""Run the 64-campaign v0.5.1 end-to-end representation benchmark pilot."""

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
from src.regression.evaluation import bemis_murcko_scaffolds
from src.regression.representation_surrogates import preprocessing_policy
from src.regression.simulation import run_regression_simulation


REPRESENTATIONS = (
    "morgan", "rdkit_descriptors", "smi_ted_light", "unimol_v1"
)
MODELS = ("random_forest", "extra_trees", "bayesian_ridge", "gaussian_process")
ACQUISITIONS = ("ucb", "ucb_qed", "ucb_diversity", "ucb_qed_diversity")
EXPECTED_DIMENSIONS = {
    "morgan": 2048,
    "rdkit_descriptors": 8,
    "smi_ted_light": 768,
    "unimol_v1": 512,
}
OUTPUT = ROOT / "results/representation_benchmark_pilot/v051"


def _validate_policy() -> dict[str, str]:
    policies = {}
    for representation in REPRESENTATIONS:
        for model in MODELS:
            policy = preprocessing_policy(model, representation)
            if representation == "morgan":
                expected = "morgan_v050"
            elif model in {"random_forest", "extra_trees"}:
                expected = "raw_dense"
            else:
                expected = "labelled_variance_filter_then_standard_scaler"
            if policy != expected:
                raise RuntimeError(
                    f"Unexpected preprocessing policy for {representation}/{model}: {policy}"
                )
            policies[f"{representation}/{model}"] = policy
    return policies


def _validate_history(history, manifest, representation, model, acquisition) -> dict:
    if len(history) != 3 or history["round"].tolist() != [0, 1, 2]:
        raise RuntimeError("Campaign history does not contain two completed acquisitions")
    if history.attrs["representation_name"] != representation:
        raise RuntimeError("Representation metadata mismatch")
    if history.attrs["model_feature_dimension"] != EXPECTED_DIMENSIONS[representation]:
        raise RuntimeError("Surrogate feature dimension mismatch")
    if history.attrs["morgan_feature_dimension"] != 2048:
        raise RuntimeError("Diversity fingerprints are not 2048D Morgan fingerprints")
    if not history.attrs["partition_manifest_used"]:
        raise RuntimeError("Campaign did not consume a fixed partition manifest")
    if history.attrs["ordered_population_hash"] != manifest.ordered_population_hash:
        raise RuntimeError("Campaign population hash mismatch")
    for attr, expected in (
        ("pool_indices", manifest.acquisition_pool_indices),
        ("test_indices", manifest.held_out_test_indices),
        ("initial_labeled_indices", manifest.initial_labeled_indices),
    ):
        np.testing.assert_array_equal(history.attrs[attr], np.asarray(expected))
    if not (history.model == model).all() or not (history.strategy == acquisition).all():
        raise RuntimeError("Campaign model/acquisition metadata mismatch")
    if not np.allclose(history.beta, 2.0):
        raise RuntimeError("Campaign beta differs from 2")

    batches = history.attrs["acquired_batches"]
    if len(batches) != 2 or [len(batch) for batch in batches] != [10, 10]:
        raise RuntimeError("Campaign did not acquire two batches of ten")
    acquired = np.concatenate(batches)
    test = set(manifest.held_out_test_indices)
    held_out = sum(int(index) in test for index in acquired)
    if len(np.unique(acquired)) != 20 or held_out:
        raise RuntimeError("Campaign acquired duplicates or held-out molecules")

    always_finite = (
        "rmse", "mae", "r2", "pearson", "best_discovered",
        "top20_mean_discovered", "mean_discovered", "dataset_best",
        "fraction_best_found", "distance_to_best", "round_runtime_seconds",
        "affinity_discovery_count", "joint_discovery_count",
        "joint_discovery_auc", "joint_scaffold_count",
    )
    for column in always_finite:
        if not np.isfinite(history[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite reported metric: {column}")
    for column in ("pool_mean_uncertainty", "pool_max_uncertainty"):
        if not np.isfinite(history[column].iloc[:2].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite pool diagnostic: {column}")
    acquisition_columns = (
        "selected_mean_prediction", "selected_mean_uncertainty",
        "selected_mean_combined_score", "selected_min_combined_score",
        "selected_max_combined_score",
    )
    for column in acquisition_columns:
        if not np.isfinite(history[column].iloc[1:].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite acquisition diagnostic: {column}")
    return {
        "rounds_completed": 2,
        "number_acquired": 20,
        "unique_acquired_count": int(len(np.unique(acquired))),
        "held_out_test_acquisitions": held_out,
        "finite_output_status": True,
        "selected_indices": json.dumps(acquired.tolist()),
    }


def main() -> None:
    policies = _validate_policy()
    representations, mask, ids, manifest, y, affinity, qed, smiles = _load_inputs()
    X_morgan = np.asarray(representations["morgan"].features[mask])
    scaffolds = bemis_murcko_scaffolds(smiles)
    records = []
    warning_records = []
    histories = []
    total_started = time.perf_counter()

    for representation in REPRESENTATIONS:
        X_model = np.asarray(representations[representation].features[mask])
        if X_model.shape != (13_983, EXPECTED_DIMENSIONS[representation]):
            raise RuntimeError(f"Unexpected common matrix shape for {representation}")
        for model in MODELS:
            for acquisition in ACQUISITIONS:
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    history = run_regression_simulation(
                        None,
                        y,
                        acquisition,
                        qed=qed,
                        affinity_nm=affinity,
                        canonical_smiles=smiles,
                        scaffolds=scaffolds,
                        model_name=model,
                        random_state=0,
                        n_initial=20,
                        batch_size=10,
                        n_rounds=2,
                        beta=2.0,
                        X_model=X_model,
                        X_morgan=X_morgan,
                        representation_name=representation,
                        partition_manifest=manifest,
                        molecule_ids=ids,
                    )
                elapsed = time.perf_counter() - started
                validation = _validate_history(
                    history, manifest, representation, model, acquisition
                )
                for item in caught:
                    warning_records.append(
                        {
                            "representation": representation,
                            "model": model,
                            "acquisition": acquisition,
                            "category": item.category.__name__,
                            "message": str(item.message),
                        }
                    )
                records.append(
                    {
                        "representation": representation,
                        "model": model,
                        "acquisition": acquisition,
                        "seed": 0,
                        **validation,
                        "runtime_seconds": elapsed,
                        "warning_count": len(caught),
                        "preprocessing_policy": policies[f"{representation}/{model}"],
                        "model_feature_dimension": X_model.shape[1],
                        "diversity_feature_dimension": X_morgan.shape[1],
                    }
                )
                persisted = history.assign(representation=representation)
                persisted.attrs = {}
                histories.append(persisted)
                print(
                    f"pilot {len(records):02d}/64: {representation} + {model} + "
                    f"{acquisition} completed in {elapsed:.2f}s; warnings={len(caught)}",
                    flush=True,
                )

    summary = pd.DataFrame(records)
    if len(summary) != 64 or not summary.finite_output_status.all():
        raise RuntimeError("Pilot matrix is incomplete or invalid")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "campaign_summary.csv", index=False)
    pd.DataFrame(warning_records).to_csv(OUTPUT / "warnings.csv", index=False)
    pd.concat(histories, ignore_index=True).to_parquet(
        OUTPUT / "campaign_histories.parquet", index=False
    )
    timing = summary.groupby(["representation", "model"], as_index=False).agg(
        campaigns=("runtime_seconds", "size"),
        total_runtime_seconds=("runtime_seconds", "sum"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        maximum_runtime_seconds=("runtime_seconds", "max"),
    )
    timing.to_csv(OUTPUT / "runtime_by_representation_model.csv", index=False)
    metadata = {
        "purpose": "integration/runtime validation only; not performance comparison",
        "campaigns": 64,
        "common_population_size": 13_983,
        "population_hash": manifest.ordered_population_hash,
        "seed": 0,
        "rounds": 2,
        "batch_size": 10,
        "beta": 2.0,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "all_campaigns_completed": True,
        "all_invariants_passed": True,
        "preprocessing_policies": policies,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
