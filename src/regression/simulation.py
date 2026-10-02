import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.regression.evaluation import (
    bemis_murcko_scaffolds,
    cumulative_curve_auc,
    evaluate_acquired_molecules,
    evaluate_regression,
)

from src.active_learning.batch_selection import TopKBatchSelector
from src.active_learning.batch_selection import BatchSelectionResult
from src.active_learning.acquisition import WeightedAcquisitionResult
from src.active_learning.criteria import AcquisitionContext
from src.active_learning.factories import (
    make_prediction_engine,
    make_ucb_engine,
    make_uncertainty_engine,
)

from src.regression.models import make_regression_surrogate
from src.regression.representation_surrogates import make_representation_surrogate
from src.data.partitions import PartitionManifest, validate_partition_manifest
from src.regression.selection import (#legacy
    #select_greedy,                    #obsolete
    select_random,
    #select_ucb,                       #obsolete
    #select_uncertainty,               #obsolete
    select_uncertainty_diverse,
    acquisition_scores,
    select_acquisition_diverse,
    select_acquisition_topk,
)


V05_ACQUISITION_STRATEGIES = {
    "ucb",
    "ucb_qed",
    "ucb_diversity",
    "ucb_qed_diversity",
}


def concatenate_simulation_histories(
    histories: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate histories without per-campaign debug provenance."""
    for history in histories:
        history.attrs.clear()

    return pd.concat(histories, ignore_index=True)


def select_with_modular_engine(
    *,
    strategy: str,
    unlabeled_indices: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_uncertainty: np.ndarray,
    X_pool: np.ndarray,
    batch_size: int,
    beta: float | None,
):
    """
    Select a batch using the modular weighted acquisition engine.

    Supported strategies:
    - greedy
    - uncertainty
    - ucb
    """
    if strategy == "greedy":
        engine = make_prediction_engine(
            normalization="robust",
        )

    elif strategy == "uncertainty":
        engine = make_uncertainty_engine(
            normalization="robust",
        )

    elif strategy == "ucb":
        if beta is None:
            raise ValueError(
                "beta must be provided when strategy='ucb'."
            )

        engine = make_ucb_engine(
            beta=beta,
            normalization="robust",
        )

    else:
        raise ValueError(
            f"Unsupported modular strategy: {strategy}"
        )

    context = AcquisitionContext(
        unlabeled_indices=unlabeled_indices,
        predicted_mean=predicted_mean,
        predicted_uncertainty=predicted_uncertainty,
        X_pool=X_pool,
    )

    acquisition_result = engine.evaluate(
        context
    )

    batch_result = TopKBatchSelector().select(
        acquisition_result,
        batch_size=batch_size,
    )

    return acquisition_result, batch_result


def run_regression_simulation(
    X: np.ndarray | None,
    y: np.ndarray,
    strategy: str,
    qed: np.ndarray | None = None,
    affinity_nm: np.ndarray | None = None,
    canonical_smiles: np.ndarray | None = None,
    scaffolds: np.ndarray | None = None,
    model_name: str = "random_forest",
    random_state: int = 42,
    n_initial: int = 20,
    batch_size: int = 10,
    n_rounds: int = 10,
    test_size: float = 0.2,
    candidate_pool_size: int = 100,
    beta: float | None = None,
    X_model: np.ndarray | None = None,
    X_morgan: np.ndarray | None = None,
    representation_name: str = "morgan",
    partition_manifest: PartitionManifest | None = None,
    molecule_ids: np.ndarray | None = None,
) -> pd.DataFrame:

    representation_aware = any(
        value is not None
        for value in (X_model, X_morgan, partition_manifest, molecule_ids)
    )
    if representation_aware:
        if X_model is None or X_morgan is None:
            raise ValueError(
                "Representation-aware simulation requires both X_model and X_morgan"
            )
        if partition_manifest is None or molecule_ids is None:
            raise ValueError(
                "Representation-aware simulation requires a fixed partition manifest "
                "and ordered molecule_ids"
            )
    else:
        if X is None:
            raise ValueError("Legacy Morgan simulation requires X")
        X_model = X
        X_morgan = X

    X_model = np.asarray(X_model)
    X_morgan = np.asarray(X_morgan)
    y = np.asarray(y)
    if X_model.ndim != 2 or X_morgan.ndim != 2:
        raise ValueError("X_model and X_morgan must be two-dimensional matrices")
    if not (len(X_model) == len(X_morgan) == len(y)):
        raise ValueError(
            "X_model, X_morgan, and y must describe the same number of molecules"
        )
    if representation_aware:
        molecule_ids = np.asarray(molecule_ids)
        if len(molecule_ids) != len(y):
            raise ValueError("Ordered molecule_ids must align with simulation matrices")
        validate_partition_manifest(partition_manifest, molecule_ids)
        if partition_manifest.random_seed != random_state:
            raise ValueError(
                "Partition-manifest seed must match the simulation random_state"
            )

    rng = np.random.default_rng(random_state)

    dataset_best = float(np.max(y))

    if strategy in V05_ACQUISITION_STRATEGIES:
        if beta is None:
            beta = 2.0

        if beta < 0:
            raise ValueError("beta must be non-negative.")

    if strategy in {"ucb_qed", "ucb_qed_diversity"}:
        if qed is None:
            raise ValueError(f"QED values are required for strategy='{strategy}'.")
        qed = np.asarray(qed, dtype=float)
        if qed.ndim != 1 or len(qed) != len(y):
            raise ValueError("QED must be one-dimensional and aligned with X and y.")

    evaluation_enabled = affinity_nm is not None or canonical_smiles is not None
    if evaluation_enabled:
        if affinity_nm is None or canonical_smiles is None or qed is None:
            raise ValueError(
                "Evaluation requires aligned affinity_nm, canonical_smiles, and QED."
            )
        affinity_nm = np.asarray(affinity_nm, dtype=float)
        canonical_smiles = np.asarray(canonical_smiles, dtype=str)
        qed = np.asarray(qed, dtype=float)
        if not (
            len(X_model) == len(y) == len(qed) == len(affinity_nm) == len(canonical_smiles)
        ):
            raise ValueError("Evaluation arrays must have identical molecular alignment.")
        if scaffolds is None:
            scaffolds = bemis_murcko_scaffolds(canonical_smiles)
        else:
            scaffolds = np.asarray(scaffolds, dtype=str)
            if len(scaffolds) != len(y):
                raise ValueError("Scaffolds must be aligned with the molecular arrays.")

    if representation_aware:
        pool_indices = np.asarray(
            partition_manifest.acquisition_pool_indices, dtype=int
        )
        test_indices = np.asarray(
            partition_manifest.held_out_test_indices, dtype=int
        )
        labeled_indices = np.asarray(
            partition_manifest.initial_labeled_indices, dtype=int
        )
    else:
        pool_indices, test_indices = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=random_state,
        )

        labeled_indices = rng.choice(
            pool_indices,
            size=min(n_initial, len(pool_indices)),
            replace=False,
        )
    initial_labeled_indices = labeled_indices.copy()

    unlabeled_indices = np.setdiff1d(
        pool_indices,
        labeled_indices,
        assume_unique=False,
    )

    # These variables describe the batch acquired after the previous round.
    # At round 0, no active-learning batch has been acquired yet.
    last_batch_mean_prediction = np.nan
    last_batch_mean_uncertainty = np.nan
    last_batch_mean_true_affinity = np.nan
    last_batch_best_true_affinity = np.nan

    # These variables are used by the new acquisition engine to track the combined score of the last batch.
    last_batch_mean_combined_score = np.nan
    last_batch_min_combined_score = np.nan
    last_batch_max_combined_score = np.nan

    last_batch_mean_prediction_contribution = np.nan
    last_batch_mean_uncertainty_contribution = np.nan

    history = []
    acquired_batches = []
    discovery_measurements = [0]
    joint_discovery_curve = [0]

    if evaluation_enabled:
        initial_endpoints = evaluate_acquired_molecules(
            acquired_indices=initial_labeled_indices,
            affinity_nm=affinity_nm,
            qed=qed,
            fingerprints=X_morgan,
            scaffolds=scaffolds,
        )

    for round_idx in range(n_rounds + 1):
        round_start = time.perf_counter()

        if representation_aware:
            model = make_representation_surrogate(
                model_name=model_name,
                representation_name=representation_name,
                random_state=random_state + round_idx,
            )
        else:
            model = make_regression_surrogate(
                model_name=model_name,
                random_state=random_state + round_idx,
            )

        model.fit(
            X_model[labeled_indices],
            y[labeled_indices],
        )

        test_predictions = model.predict(
            X_model[test_indices]
        )

        metrics = evaluate_regression(
            y[test_indices],
            test_predictions,
        )

        discovered = y[labeled_indices]

        history_row = {
                "round": round_idx,
                "strategy": strategy,
                "model": model_name,
                "beta": (
                    beta
                    if strategy in V05_ACQUISITION_STRATEGIES
                    else np.nan
                ),
                "seed": random_state,
                "n_labeled": len(labeled_indices),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "pearson": metrics["pearson"],
                "best_discovered": float(
                    discovered.max()
                ),
                "top20_mean_discovered": float(
                    np.mean(
                        np.sort(discovered)[
                            -min(20, len(discovered)):
                        ]
                    )
                ),
                "mean_discovered": float(
                    discovered.mean()
                ),
                "dataset_best": dataset_best,
                "fraction_best_found": float(
                    discovered.max() / dataset_best
                ) if dataset_best != 0 else np.nan,
                "distance_to_best": float(
                    dataset_best - discovered.max()
                ),
                "round_runtime_seconds": np.nan,
                "pool_mean_uncertainty": np.nan,
                "pool_max_uncertainty": np.nan,
                "selected_mean_prediction": (
                    last_batch_mean_prediction
                ),
                "selected_mean_uncertainty": (
                    last_batch_mean_uncertainty
                ),
                "selected_mean_true_affinity": (
                    last_batch_mean_true_affinity
                ),
                "selected_best_true_affinity": (
                    last_batch_best_true_affinity
                ),
                # new engine metrics
                "selected_mean_combined_score": (
                    last_batch_mean_combined_score
                ),
                "selected_min_combined_score": (
                    last_batch_min_combined_score
                ),
                "selected_max_combined_score": (
                    last_batch_max_combined_score
                ),
                "selected_mean_prediction_contribution": (
                    last_batch_mean_prediction_contribution
                ),
                "selected_mean_uncertainty_contribution": (
                    last_batch_mean_uncertainty_contribution
                ),
            }

        if evaluation_enabled:
            newly_acquired = (
                acquired_batches[-1]
                if acquired_batches
                else np.array([], dtype=int)
            )
            cumulative_acquired = (
                np.concatenate(acquired_batches)
                if acquired_batches
                else np.array([], dtype=int)
            )
            endpoints = evaluate_acquired_molecules(
                acquired_indices=cumulative_acquired,
                affinity_nm=affinity_nm,
                qed=qed,
                fingerprints=X_morgan,
                scaffolds=scaffolds,
            )
            if round_idx > 0:
                discovery_measurements.append(len(cumulative_acquired))
                joint_discovery_curve.append(endpoints["joint_discovery_count"])

            history_row.update(
                {
                    "n_initial_labeled": len(initial_labeled_indices),
                    "n_newly_acquired": len(newly_acquired),
                    "n_cumulative_acquired": len(cumulative_acquired),
                    "initial_affinity_discovery_count": initial_endpoints[
                        "affinity_discovery_count"
                    ],
                    "initial_joint_discovery_count": initial_endpoints[
                        "joint_discovery_count"
                    ],
                    "affinity_discovery_count": endpoints[
                        "affinity_discovery_count"
                    ],
                    "joint_discovery_count": endpoints["joint_discovery_count"],
                    "joint_discovery_auc": cumulative_curve_auc(
                        discovery_measurements,
                        joint_discovery_curve,
                    ),
                    "joint_scaffold_count": endpoints["joint_scaffold_count"],
                    "joint_diversity": endpoints["joint_diversity"],
                    "initial_labeled_indices": json.dumps(
                        initial_labeled_indices.tolist()
                    ),
                    "newly_acquired_indices": json.dumps(newly_acquired.tolist()),
                    "cumulative_acquired_indices": json.dumps(
                        cumulative_acquired.tolist()
                    ),
                    "newly_acquired_smiles": json.dumps(
                        canonical_smiles[newly_acquired].tolist()
                    ),
                    "newly_acquired_affinity_nm": json.dumps(
                        affinity_nm[newly_acquired].tolist()
                    ),
                    "newly_acquired_qed": json.dumps(qed[newly_acquired].tolist()),
                    "joint_qualifying_indices": json.dumps(
                        endpoints["joint_qualifying_indices"].tolist()
                    ),
                }
            )

        history.append(history_row)

        if (
            round_idx == n_rounds
            or len(unlabeled_indices) == 0
        ):
            history[-1]["round_runtime_seconds"] = float(
                time.perf_counter() - round_start
            )
            break

        predicted_mean, predicted_uncertainty = (
            model.predict_with_uncertainty(
                X_model[unlabeled_indices],
            )
        )

        history[-1]["pool_mean_uncertainty"] = float(
            np.mean(predicted_uncertainty)
        )
        history[-1]["pool_max_uncertainty"] = float(
            np.max(predicted_uncertainty)
        )

        acquisition_result = None
        batch_result = None

        if strategy == "random":
            selected = select_random(
                unlabeled_indices,
                batch_size,
                rng,
            )

        elif strategy in {
            "greedy",
            "uncertainty",
        }:
            acquisition_result, batch_result = (
                select_with_modular_engine(
                    strategy=strategy,
                    unlabeled_indices=unlabeled_indices,
                    predicted_mean=predicted_mean,
                    predicted_uncertainty=predicted_uncertainty,
                    X_pool=X_model[unlabeled_indices],
                    batch_size=batch_size,
                    beta=beta,
                )
            )

            selected = batch_result.selected_indices

        elif strategy in V05_ACQUISITION_STRATEGIES:
            pool_qed = (
                qed[unlabeled_indices]
                if strategy in {"ucb_qed", "ucb_qed_diversity"}
                else None
            )
            scores = acquisition_scores(
                predicted_mean,
                predicted_uncertainty,
                qed=pool_qed,
                beta=beta,
            )
            base_result = make_ucb_engine(
                beta=beta,
                normalization="robust",
            ).evaluate(
                AcquisitionContext(
                    unlabeled_indices=unlabeled_indices,
                    predicted_mean=predicted_mean,
                    predicted_uncertainty=predicted_uncertainty,
                    X_pool=X_model[unlabeled_indices],
                )
            )
            acquisition_result = WeightedAcquisitionResult(
                unlabeled_indices=base_result.unlabeled_indices,
                combined_scores=scores,
                criterion_evaluations=base_result.criterion_evaluations,
            )

            if strategy in {"ucb_diversity", "ucb_qed_diversity"}:
                selected = select_acquisition_diverse(
                    X=X_morgan,
                    unlabeled_indices=unlabeled_indices,
                    labeled_indices=labeled_indices,
                    scores=scores,
                    batch_size=batch_size,
                    candidate_pool_size=50,
                )
                selected_positions = np.asarray(
                    [
                        np.flatnonzero(unlabeled_indices == index)[0]
                        for index in selected
                    ],
                    dtype=int,
                )
                batch_result = BatchSelectionResult(
                    selected_indices=selected,
                    selected_pool_positions=selected_positions,
                    selected_combined_scores=scores[selected_positions],
                )
            else:
                batch_result = TopKBatchSelector().select(
                    acquisition_result,
                    batch_size=batch_size,
                )
                selected = batch_result.selected_indices

        elif strategy == "uncertainty_diverse":
            selected = select_uncertainty_diverse(
                X_morgan,
                unlabeled_indices,
                predicted_uncertainty,
                batch_size,
                candidate_pool_size,
                random_state=random_state + round_idx,
            )

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}"
            )        
        

        if acquisition_result is not None:
            selected_positions = (
                batch_result.selected_pool_positions
            )

            selected_combined_scores = (
                acquisition_result.combined_scores[
                    selected_positions
                ]
            )

            last_batch_mean_combined_score = float(
                np.mean(selected_combined_scores)
            )

            last_batch_min_combined_score = float(
                np.min(selected_combined_scores)
            )

            last_batch_max_combined_score = float(
                np.max(selected_combined_scores)
            )

            if (
                "prediction"
                in acquisition_result.criterion_evaluations
            ):
                prediction_evaluation = (
                    acquisition_result
                    .criterion_evaluations["prediction"]
                )

                last_batch_mean_prediction_contribution = (
                    float(
                        np.mean(
                            prediction_evaluation
                            .weighted_contribution[
                                selected_positions
                            ]
                        )
                    )
                )
            else:
                last_batch_mean_prediction_contribution = (
                    np.nan
                )

            if (
                "uncertainty"
                in acquisition_result.criterion_evaluations
            ):
                uncertainty_evaluation = (
                    acquisition_result
                    .criterion_evaluations["uncertainty"]
                )

                last_batch_mean_uncertainty_contribution = (
                    float(
                        np.mean(
                            uncertainty_evaluation
                            .weighted_contribution[
                                selected_positions
                            ]
                        )
                    )
                )
            else:
                last_batch_mean_uncertainty_contribution = (
                    np.nan
                )

        else:
            last_batch_mean_combined_score = np.nan
            last_batch_min_combined_score = np.nan
            last_batch_max_combined_score = np.nan
            last_batch_mean_prediction_contribution = np.nan
            last_batch_mean_uncertainty_contribution = np.nan

        # Map selected global indices back to their positions in the
        # current unlabeled pool so that prediction and uncertainty
        # diagnostics can be extracted.
        selected_mask = np.isin(
            unlabeled_indices,
            selected,
        )

        if selected_mask.sum() != len(selected):
            raise RuntimeError(
                "Could not map all selected indices to the unlabeled pool."
            )

        selected_predicted_mean = (
            predicted_mean[selected_mask]
        )

        selected_uncertainty_values = (
            predicted_uncertainty[selected_mask]
        )

        selected_true_affinity = y[selected]
        acquired_batches.append(selected.copy())

        # These values will be written into the next history row.
        last_batch_mean_prediction = float(
            np.mean(selected_predicted_mean)
        )

        last_batch_mean_uncertainty = float(
            np.mean(selected_uncertainty_values)
        )

        last_batch_mean_true_affinity = float(
            np.mean(selected_true_affinity)
        )

        last_batch_best_true_affinity = float(
            np.max(selected_true_affinity)
        )

        labeled_indices = np.concatenate(
            [labeled_indices, selected]
        )

        unlabeled_indices = np.setdiff1d(
            unlabeled_indices,
            selected,
            assume_unique=False,
        )

        history[-1]["round_runtime_seconds"] = float(
            time.perf_counter() - round_start
        )

    result = pd.DataFrame(history)
    # Test/debug provenance without changing the persisted history schema.
    result.attrs["pool_indices"] = pool_indices.copy()
    result.attrs["test_indices"] = test_indices.copy()
    result.attrs["initial_labeled_indices"] = initial_labeled_indices
    result.attrs["acquired_batches"] = acquired_batches
    result.attrs["representation_name"] = representation_name
    result.attrs["model_feature_dimension"] = X_model.shape[1]
    result.attrs["morgan_feature_dimension"] = X_morgan.shape[1]
    result.attrs["partition_manifest_used"] = representation_aware
    result.attrs["ordered_population_hash"] = (
        partition_manifest.ordered_population_hash if representation_aware else None
    )
    return result
