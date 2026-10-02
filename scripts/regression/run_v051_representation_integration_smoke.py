#!/usr/bin/env python3
"""Run the small fixed-partition v0.5.1 representation integration smoke test."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.partitions import load_partition_manifest
from src.data.representations import (
    load_frozen_embedding_representation,
    load_morgan_representation,
    load_rdkit_descriptor_representation,
    ordered_population_hash,
)
from src.regression.evaluation import bemis_murcko_scaffolds
from src.regression.simulation import run_regression_simulation


TARGET_DIR = REPOSITORY_ROOT / "data/processed/v051/targets/EGFR"
REPRESENTATION_DIR = REPOSITORY_ROOT / "data/processed/v051/representations"
COMMON_DIR = REPOSITORY_ROOT / "data/processed/v051/common_population"
OUTPUT_DIR = REPOSITORY_ROOT / "results/representation_integration/v051"
EXPECTED_COMMON_SIZE = 13_983

SMOKE_RUNS = (
    ("morgan", "random_forest", "ucb"),
    ("rdkit_descriptors", "random_forest", "ucb"),
    ("rdkit_descriptors", "bayesian_ridge", "ucb"),
    ("smi_ted_light", "bayesian_ridge", "ucb"),
    ("unimol_v1", "gaussian_process", "ucb"),
    ("smi_ted_light", "bayesian_ridge", "ucb_diversity"),
)


def _load_inputs():
    base_dir = TARGET_DIR.parent
    morgan = load_morgan_representation("EGFR", base_dir)
    representations = {
        "morgan": morgan,
        "rdkit_descriptors": load_rdkit_descriptor_representation(
            "EGFR", base_dir, reference=morgan
        ),
        "smi_ted_light": load_frozen_embedding_representation(
            "smi_ted_light",
            REPRESENTATION_DIR / "smi_ted_light",
            expected_dimension=768,
            reference=morgan,
        ),
        "unimol_v1": load_frozen_embedding_representation(
            "unimol_v1",
            REPRESENTATION_DIR / "unimol_v1",
            expected_dimension=512,
            reference=morgan,
        ),
    }
    mask = np.load(COMMON_DIR / "common_mask.npy").astype(bool, copy=False)
    common_ids = np.load(COMMON_DIR / "molecule_ids.npy").astype(str)
    original_indices = np.load(COMMON_DIR / "original_indices.npy")
    if len(common_ids) != EXPECTED_COMMON_SIZE or int(mask.sum()) != EXPECTED_COMMON_SIZE:
        raise ValueError("Unexpected common-population size")
    if not np.array_equal(np.flatnonzero(mask), original_indices):
        raise ValueError("Common source-index mapping does not match common mask")
    if not np.array_equal(morgan.molecule_ids[mask], common_ids):
        raise ValueError("Common molecular identities do not match Morgan source order")
    for name, representation in representations.items():
        if not np.array_equal(representation.molecule_ids[mask], common_ids):
            raise ValueError(f"Common molecular identities do not match {name}")
        if not representation.success[mask].all():
            raise ValueError(f"Common population contains an ineligible {name} row")

    manifest = load_partition_manifest(
        COMMON_DIR / "partitions/seed_00.json", common_ids
    )
    clean = pd.read_parquet(TARGET_DIR / "clean.parquet")
    descriptors = pd.read_parquet(TARGET_DIR / "descriptors.parquet")
    y = np.load(TARGET_DIR / "affinity.npy")[mask]
    affinity_nm = clean.aff_nM_median.to_numpy(dtype=float)[mask]
    smiles = clean.canonical_smiles.astype(str).to_numpy()[mask]
    qed = descriptors.QED.to_numpy(dtype=float)[mask]
    if not np.array_equal(smiles, morgan.canonical_smiles[mask]):
        raise ValueError("Corrected target rows do not match common representation order")
    if not all(len(values) == EXPECTED_COMMON_SIZE for values in (y, affinity_nm, qed)):
        raise ValueError("Common target arrays are not aligned")
    return representations, mask, common_ids, manifest, y, affinity_nm, qed, smiles


def main() -> None:
    representations, mask, ids, manifest, y, affinity_nm, qed, smiles = _load_inputs()
    X_morgan = np.asarray(representations["morgan"].features[mask])
    if X_morgan.shape != (EXPECTED_COMMON_SIZE, 2048):
        raise ValueError("Common Morgan matrix has an unexpected shape")
    scaffolds = bemis_murcko_scaffolds(smiles)
    summaries = []
    histories = []
    test_set = set(manifest.held_out_test_indices)

    for representation_name, model_name, strategy in SMOKE_RUNS:
        X_model = np.asarray(representations[representation_name].features[mask])
        started = time.perf_counter()
        history = run_regression_simulation(
            None,
            y,
            strategy,
            qed=qed,
            affinity_nm=affinity_nm,
            canonical_smiles=smiles,
            scaffolds=scaffolds,
            model_name=model_name,
            random_state=0,
            n_initial=20,
            batch_size=10,
            n_rounds=2,
            beta=2.0,
            X_model=X_model,
            X_morgan=X_morgan,
            representation_name=representation_name,
            partition_manifest=manifest,
            molecule_ids=ids,
        )
        elapsed = time.perf_counter() - started
        acquired = np.concatenate(history.attrs["acquired_batches"])
        if len(history) != 3 or [len(x) for x in history.attrs["acquired_batches"]] != [10, 10]:
            raise RuntimeError("Smoke run did not complete two ten-molecule rounds")
        if len(np.unique(acquired)) != 20 or any(int(x) in test_set for x in acquired):
            raise RuntimeError("Smoke acquisition indices are duplicated or enter the test set")
        if not np.array_equal(
            history.attrs["initial_labeled_indices"],
            np.asarray(manifest.initial_labeled_indices),
        ):
            raise RuntimeError("Smoke run did not use the fixed initial labelled set")
        finite_columns = [
            "rmse", "mae", "r2", "best_discovered", "pool_mean_uncertainty",
            "pool_max_uncertainty",
        ]
        for column in finite_columns:
            values = history[column].to_numpy(dtype=float)
            if column.startswith("pool_"):
                values = values[:-1]
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite smoke output in {column}")
        persisted_history = history.assign(
            representation_name=representation_name, smoke_model=model_name
        )
        # Simulation attrs contain aligned NumPy arrays for in-memory validation.
        # Pandas compares attrs during concat, where array equality is ambiguous.
        persisted_history.attrs = {}
        histories.append(persisted_history)
        summaries.append(
            {
                "representation": representation_name,
                "model": model_name,
                "strategy": strategy,
                "model_feature_dimension": int(X_model.shape[1]),
                "diversity_feature_dimension": int(X_morgan.shape[1]),
                "rounds_completed": 2,
                "acquired_count": int(len(acquired)),
                "runtime_seconds": elapsed,
                "completed": True,
            }
        )
        print(
            f"Smoke: {representation_name} + {model_name} + {strategy} "
            f"completed in {elapsed:.2f}s",
            flush=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(histories, ignore_index=True).to_parquet(
        OUTPUT_DIR / "smoke_histories.parquet", index=False
    )
    summary = {
        "purpose": "integration validation only; not a representation comparison",
        "common_population_size": EXPECTED_COMMON_SIZE,
        "ordered_population_hash": ordered_population_hash(ids),
        "partition_seed": 0,
        "initial_labeled_count": 20,
        "rounds": 2,
        "batch_size": 10,
        "runs": summaries,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# v0.5.1 representation integration smoke test",
        "",
        "Infrastructure validation only; the outputs are not a scientific representation comparison.",
        "",
        f"- Common population: {EXPECTED_COMMON_SIZE}",
        f"- Fixed partition: seed 0 ({summary['ordered_population_hash']})",
        "- Initial labelled / rounds / batch size: 20 / 2 / 10",
        "",
    ]
    lines.extend(
        f"- {run['representation']} + {run['model']} + {run['strategy']}: "
        f"completed, {run['runtime_seconds']:.2f}s"
        for run in summaries
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
