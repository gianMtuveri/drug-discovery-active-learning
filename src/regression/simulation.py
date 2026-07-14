import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.regression.evaluation import evaluate_regression
from src.regression.models import make_regression_model
from src.regression.selection import (
    select_greedy,
    select_random,
    select_ucb,
    select_uncertainty,
    select_uncertainty_diverse,
)
from src.regression.uncertainty import predict_with_uncertainty


def run_regression_simulation(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    random_state: int = 42,
    n_initial: int = 20,
    batch_size: int = 10,
    n_rounds: int = 10,
    test_size: float = 0.2,
    candidate_pool_size: int = 100,
    beta: float | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    if strategy == "ucb":
        if beta is None:
            raise ValueError(
                "beta must be provided when strategy='ucb'."
            )

        if beta < 0:
            raise ValueError("beta must be non-negative.")

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

    history = []

    for round_idx in range(n_rounds + 1):
        model = make_regression_model(
            model_name="random_forest",
            random_state=random_state + round_idx,
        )

        model.fit(
            X[labeled_indices],
            y[labeled_indices],
        )

        test_predictions = model.predict(
            X[test_indices]
        )

        metrics = evaluate_regression(
            y[test_indices],
            test_predictions,
        )

        discovered = y[labeled_indices]

        history.append(
            {
                "round": round_idx,
                "strategy": strategy,
                "beta": (
                    beta
                    if strategy == "ucb"
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
            }
        )

        if (
            round_idx == n_rounds
            or len(unlabeled_indices) == 0
        ):
            break

        predicted_mean, predicted_uncertainty = (
            predict_with_uncertainty(
                model,
                X[unlabeled_indices],
            )
        )

        if strategy == "random":
            selected = select_random(
                unlabeled_indices,
                batch_size,
                rng,
            )

        elif strategy == "greedy":
            selected = select_greedy(
                unlabeled_indices,
                predicted_mean,
                batch_size,
            )

        elif strategy == "uncertainty":
            selected = select_uncertainty(
                unlabeled_indices,
                predicted_uncertainty,
                batch_size,
            )

        elif strategy == "uncertainty_diverse":
            selected = select_uncertainty_diverse(
                X,
                unlabeled_indices,
                predicted_uncertainty,
                batch_size,
                candidate_pool_size,
                random_state=random_state + round_idx,
            )

        elif strategy == "ucb":
            selected = select_ucb(
                unlabeled_indices=unlabeled_indices,
                predicted_affinity=predicted_mean,
                uncertainty=predicted_uncertainty,
                batch_size=batch_size,
                beta=beta,
            )

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

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

    return pd.DataFrame(history)