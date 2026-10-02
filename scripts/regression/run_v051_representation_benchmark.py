#!/usr/bin/env python3
"""Run the frozen 960-campaign v0.5.1 representation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
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
from src.regression.evaluation import bemis_murcko_scaffolds
from src.regression.representation_surrogates import preprocessing_policy
from src.regression.simulation import run_regression_simulation


REPRESENTATIONS = (
    "morgan", "rdkit_descriptors", "smi_ted_light", "unimol_v1"
)
MODELS = ("random_forest", "bayesian_ridge", "gaussian_process")
ACQUISITIONS = ("ucb", "ucb_qed", "ucb_diversity", "ucb_qed_diversity")
SEEDS = tuple(range(20))
EXPECTED_CAMPAIGNS = 4 * 3 * 4 * 20
EXPECTED_COMMON_SIZE = 13_983
EXPECTED_DIMENSIONS = {
    "morgan": 2048,
    "rdkit_descriptors": 8,
    "smi_ted_light": 768,
    "unimol_v1": 512,
}
COMMON_DIR = ROOT / "data/processed/v051/common_population"
DEFAULT_OUTPUT = ROOT / "results/representation_benchmark/v051"
SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class CampaignSpec:
    representation: str
    model: str
    acquisition: str
    seed: int

    @property
    def identifier(self) -> str:
        return (
            f"{self.representation}__{self.model}__"
            f"{self.acquisition}__seed_{self.seed:02d}"
        )


def campaign_matrix() -> tuple[CampaignSpec, ...]:
    matrix = tuple(
        CampaignSpec(representation, model, acquisition, seed)
        for representation in REPRESENTATIONS
        for model in MODELS
        for acquisition in ACQUISITIONS
        for seed in SEEDS
    )
    if len(matrix) != EXPECTED_CAMPAIGNS or len(set(matrix)) != EXPECTED_CAMPAIGNS:
        raise RuntimeError(
            f"Benchmark matrix must contain exactly {EXPECTED_CAMPAIGNS} campaigns"
        )
    return matrix


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _expected_policy(spec: CampaignSpec) -> str:
    policy = preprocessing_policy(spec.model, spec.representation)
    expected = (
        "morgan_v050"
        if spec.representation == "morgan"
        else "raw_dense"
        if spec.model == "random_forest"
        else "labelled_variance_filter_then_standard_scaler"
    )
    if policy != expected:
        raise RuntimeError(f"Unexpected preprocessing policy for {spec}: {policy}")
    return policy


def validate_campaign_history(
    history: pd.DataFrame,
    spec: CampaignSpec,
    *,
    population_hash: str,
    held_out_test_indices: tuple[int, ...],
    initial_labeled_indices: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Validate one persisted or in-memory completed campaign."""
    if len(history) != 21 or history["round"].tolist() != list(range(21)):
        raise ValueError("Campaign must contain history rounds 0..20")
    exact_columns = {
        "representation": spec.representation,
        "model": spec.model,
        "strategy": spec.acquisition,
        "seed": spec.seed,
        "population_hash": population_hash,
        "preprocessing_policy": _expected_policy(spec),
        "model_feature_dimension": EXPECTED_DIMENSIONS[spec.representation],
        "morgan_feature_dimension": 2048,
    }
    for column, expected in exact_columns.items():
        if column not in history or not (history[column] == expected).all():
            raise ValueError(f"Campaign metadata mismatch: {column}")
    if not np.allclose(history["beta"].to_numpy(dtype=float), 2.0):
        raise ValueError("Campaign beta differs from 2")
    if history["n_newly_acquired"].tolist() != [0] + [10] * 20:
        raise ValueError("Campaign does not contain twenty ten-molecule batches")
    if history["n_cumulative_acquired"].tolist() != list(range(0, 201, 10)):
        raise ValueError("Campaign cumulative acquisition counts are invalid")
    if initial_labeled_indices is not None:
        observed_initial = np.asarray(
            json.loads(history["initial_labeled_indices"].iloc[0]), dtype=int
        )
        if not np.array_equal(observed_initial, np.asarray(initial_labeled_indices)):
            raise ValueError("Campaign initial labelled identities differ from manifest")

    batches = [
        json.loads(value) for value in history["newly_acquired_indices"].iloc[1:]
    ]
    if len(batches) != 20 or any(len(batch) != 10 for batch in batches):
        raise ValueError("Campaign acquisition batch serialization is invalid")
    acquired = np.asarray([value for batch in batches for value in batch], dtype=int)
    if len(acquired) != 200 or len(np.unique(acquired)) != 200:
        raise ValueError("Campaign must acquire 200 unique molecules")
    held_out = np.asarray(held_out_test_indices, dtype=int)
    if np.isin(acquired, held_out).any():
        raise ValueError("Campaign acquired a held-out test molecule")

    always_finite = (
        "rmse", "mae", "r2", "best_discovered",
        "top20_mean_discovered", "mean_discovered", "dataset_best",
        "fraction_best_found", "distance_to_best", "round_runtime_seconds",
        "affinity_discovery_count", "joint_discovery_count",
        "joint_discovery_auc", "joint_scaffold_count",
    )
    for column in always_finite:
        if not np.isfinite(history[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite campaign output: {column}")
    # Pearson correlation is mathematically undefined when either input is
    # constant.  SciPy reports that condition as NaN plus ConstantInputWarning;
    # it is not a non-finite prediction or acquisition output.  Preserve the
    # undefined value, but reject infinities.
    pearson = history["pearson"].to_numpy(dtype=float)
    if np.isinf(pearson).any():
        raise ValueError("Infinite campaign output: pearson")
    for column in ("pool_mean_uncertainty", "pool_max_uncertainty"):
        if not np.isfinite(history[column].iloc[:-1].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite pool diagnostic: {column}")
    for column in (
        "selected_mean_prediction", "selected_mean_uncertainty",
        "selected_mean_combined_score", "selected_min_combined_score",
        "selected_max_combined_score",
    ):
        if not np.isfinite(history[column].iloc[1:].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite acquisition diagnostic: {column}")
    return {
        "campaign_id": spec.identifier,
        **asdict(spec),
        "rounds_completed": 20,
        "acquired_count": 200,
        "unique_acquired_count": int(len(np.unique(acquired))),
        "held_out_test_acquisitions": 0,
        "undefined_pearson_rounds": int(np.isnan(pearson).sum()),
        "valid": True,
    }


def _persistable_history(
    history: pd.DataFrame,
    spec: CampaignSpec,
    *,
    population_hash: str,
    policy: str,
    runtime_seconds: float,
) -> pd.DataFrame:
    persisted = history.assign(
        representation=spec.representation,
        population_hash=population_hash,
        preprocessing_policy=policy,
        model_feature_dimension=EXPECTED_DIMENSIONS[spec.representation],
        morgan_feature_dimension=2048,
        campaign_id=spec.identifier,
        campaign_runtime_seconds=float(runtime_seconds),
    )
    persisted.attrs = {}
    return persisted


def _load_valid_campaign(
    path: Path, warning_path: Path, spec: CampaignSpec, population_hash: str,
    test: tuple[int, ...], initial: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    if not path.is_file() or not warning_path.is_file():
        return None
    try:
        history = pd.read_parquet(path)
        warning_payload = json.loads(warning_path.read_text())
        if not isinstance(warning_payload, list):
            raise ValueError("Campaign warning checkpoint is not a JSON list")
        validation = validate_campaign_history(
            history, spec, population_hash=population_hash,
            held_out_test_indices=test, initial_labeled_indices=initial,
        )
    except Exception as exc:
        print(f"Invalid checkpoint will be replaced: {path.name}: {exc}", flush=True)
        return None
    return history, validation


def _write_final_outputs(output: Path, matrix: tuple[CampaignSpec, ...], wall: float) -> None:
    histories: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for spec in matrix:
        manifest = json.loads(
            (COMMON_DIR / "common_population_manifest.json").read_text()
        )
        ids = np.load(COMMON_DIR / "molecule_ids.npy").astype(str)
        partition = load_partition_manifest(
            COMMON_DIR / f"partitions/seed_{spec.seed:02d}.json", ids
        )
        path = output / "campaigns" / f"{spec.identifier}.parquet"
        history = pd.read_parquet(path)
        validation = validate_campaign_history(
            history, spec,
            population_hash=manifest["common_ordered_population_hash"],
            held_out_test_indices=partition.held_out_test_indices,
            initial_labeled_indices=partition.initial_labeled_indices,
        )
        histories.append(history)
        warning_path = output / "campaign_warnings" / f"{spec.identifier}.json"
        campaign_warnings = json.loads(warning_path.read_text())
        warning_rows.extend(campaign_warnings)
        summaries.append({
            **validation,
            "runtime_seconds": float(history["campaign_runtime_seconds"].iloc[0]),
            "warning_count": len(campaign_warnings),
        })

    combined = pd.concat(histories, ignore_index=True)
    summary = pd.DataFrame(summaries)
    warnings_frame = pd.DataFrame(
        warning_rows,
        columns=("campaign_id", "representation", "model", "acquisition", "seed",
                 "category", "message"),
    )
    _atomic_parquet(combined, output / "combined_campaign_history.parquet")
    _atomic_parquet(summary, output / "campaign_summary.parquet")
    _atomic_parquet(warnings_frame, output / "warnings.parquet")
    summary.to_csv(output / "campaign_summary.csv", index=False)
    warnings_frame.to_csv(output / "warnings.csv", index=False)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "campaigns_expected": EXPECTED_CAMPAIGNS,
        "campaigns_completed": len(summary),
        "campaigns_failed": 0,
        "history_rows_expected": EXPECTED_CAMPAIGNS * 21,
        "history_rows_observed": len(combined),
        "all_campaign_invariants_passed": bool(summary["valid"].all()),
        "total_wall_time_seconds_this_invocation": wall,
    }
    _atomic_json(validation, output / "validation_summary.json")


def run(output: Path) -> None:
    matrix = campaign_matrix()
    print(
        f"Validated benchmark matrix: 4 * 3 * 4 * 20 = {len(matrix)} campaigns",
        flush=True,
    )
    representations, mask, ids, _, y, affinity, qed, smiles = _load_inputs()
    if len(ids) != EXPECTED_COMMON_SIZE:
        raise RuntimeError("Frozen common population does not contain 13,983 rows")
    population_manifest = json.loads(
        (COMMON_DIR / "common_population_manifest.json").read_text()
    )
    population_hash = population_manifest["common_ordered_population_hash"]
    X_morgan = np.asarray(representations["morgan"].features[mask])
    if X_morgan.shape != (EXPECTED_COMMON_SIZE, 2048):
        raise RuntimeError("Frozen common Morgan matrix has an unexpected shape")
    scaffolds = bemis_murcko_scaffolds(smiles)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "v0.5.1 molecular-representation active-learning benchmark",
        "campaigns": EXPECTED_CAMPAIGNS,
        "matrix_formula": "4 representations * 3 models * 4 acquisitions * 20 seeds",
        "representations": list(REPRESENTATIONS),
        "models": list(MODELS),
        "extra_trees_excluded": True,
        "extra_trees_exclusion_note": (
            "Extra Trees remains supported by the codebase but is excluded from "
            "the v0.5.1 scientific benchmark."
        ),
        "acquisitions": list(ACQUISITIONS),
        "seeds": list(SEEDS),
        "common_population_size": EXPECTED_COMMON_SIZE,
        "common_population_hash": population_hash,
        "initial_labelled": 20,
        "rounds": 20,
        "batch_size": 10,
        "beta": 2.0,
        "diversity_representation": "2048-bit Morgan fingerprints",
        "checkpointing": "one atomically written validated parquet per campaign",
    }
    _atomic_json(benchmark_manifest, output / "benchmark_manifest.json")

    started_all = time.perf_counter()
    completed = 0
    skipped = 0
    for spec in matrix:
        partition = load_partition_manifest(
            COMMON_DIR / f"partitions/seed_{spec.seed:02d}.json", ids
        )
        campaign_path = output / "campaigns" / f"{spec.identifier}.parquet"
        warning_path = output / "campaign_warnings" / f"{spec.identifier}.json"
        existing = _load_valid_campaign(
            campaign_path, warning_path, spec, population_hash,
            partition.held_out_test_indices,
            partition.initial_labeled_indices,
        )
        if existing is not None:
            completed += 1
            skipped += 1
            if completed % 20 == 0 or completed == EXPECTED_CAMPAIGNS:
                print(
                    f"benchmark {completed}/{EXPECTED_CAMPAIGNS}: valid checkpoint "
                    f"({skipped} skipped)", flush=True,
                )
            continue

        X_model = np.asarray(representations[spec.representation].features[mask])
        expected_shape = (EXPECTED_COMMON_SIZE, EXPECTED_DIMENSIONS[spec.representation])
        if X_model.shape != expected_shape:
            raise RuntimeError(f"Unexpected matrix shape for {spec.representation}")
        policy = _expected_policy(spec)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            history = run_regression_simulation(
                None, y, spec.acquisition, qed=qed, affinity_nm=affinity,
                canonical_smiles=smiles, scaffolds=scaffolds, model_name=spec.model,
                random_state=spec.seed, n_initial=20, batch_size=10, n_rounds=20,
                beta=2.0, X_model=X_model, X_morgan=X_morgan,
                representation_name=spec.representation,
                partition_manifest=partition, molecule_ids=ids,
            )
        elapsed = time.perf_counter() - started
        persisted = _persistable_history(
            history, spec, population_hash=population_hash, policy=policy,
            runtime_seconds=elapsed,
        )
        validate_campaign_history(
            persisted, spec, population_hash=population_hash,
            held_out_test_indices=partition.held_out_test_indices,
            initial_labeled_indices=partition.initial_labeled_indices,
        )
        warning_payload = [
            {
                "campaign_id": spec.identifier,
                **asdict(spec),
                "category": item.category.__name__,
                "message": str(item.message),
            }
            for item in caught
        ]
        _atomic_parquet(persisted, campaign_path)
        _atomic_json(
            warning_payload,
            warning_path,
        )
        completed += 1
        print(
            f"benchmark {completed}/{EXPECTED_CAMPAIGNS}: {spec.identifier} "
            f"completed in {elapsed:.2f}s; warnings={len(caught)}",
            flush=True,
        )

    wall = time.perf_counter() - started_all
    _write_final_outputs(output, matrix, wall)
    print(
        f"Benchmark complete: {completed}/{EXPECTED_CAMPAIGNS}; "
        f"this invocation {wall:.2f}s; resumed={skipped}", flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output_dir.resolve())
