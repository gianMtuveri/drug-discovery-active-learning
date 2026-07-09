import pandas as pd
from sklearn.model_selection import train_test_split

from src.pool import initialize_pool, update_pool
from src.model import train_model
from src.selection import (
    select_random,
    select_greedy,
    select_uncertainty_topk,
    select_uncertainty_diverse,
    select_query_by_committee,
)
from src.evaluation import evaluate_model
from src.committee import train_committee, compute_disagreement_scores


def run_simulation(
    X,
    y,
    strategy,
    initialization_strategy="random",
    n_initial=20,
    batch_size=10,
    n_rounds=10,
    test_size=0.2,
    random_state=42,
):
    """
    Run one active-learning simulation.

    Scientific meaning
    ------------------
    Simulate an experimental campaign where a model is repeatedly trained
    on currently tested molecules and used to choose the next molecules
    to test.

    Parameters
    ----------
    X : np.ndarray
        Features for all molecules.
    y : np.ndarray
        Binary labels for all molecules.
    strategy : str
        Selection strategy: 'random', 'greedy', or 'uncertainty'.
    n_initial : int
        Number of molecules tested before active learning starts.
    batch_size : int
        Number of new experiments per round.
    n_rounds : int
        Number of active-learning rounds.
    test_size : float
        Fraction of molecules reserved as fixed test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    history : pd.DataFrame
        One row per round, containing model performance and discovery metrics.
    """

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    labeled_indices, unlabeled_indices = initialize_pool(
        X=X_pool,
        n_initial=n_initial,
        strategy=initialization_strategy,
        random_state=random_state,
    )

    initial_indices = labeled_indices.copy()

    history = []

    for round_id in range(n_rounds + 1):
        model = train_model(X_pool, y_pool, labeled_indices)

        metrics = evaluate_model(model, X_test, y_test)

        n_actives_found = int(y_pool[labeled_indices].sum())

        history.append(
        {
            "round": round_id,
            "strategy": strategy,
            "n_labeled": len(labeled_indices),
            "n_actives_found": n_actives_found,
            "roc_auc": metrics["roc_auc"],
            "initial_indices": initial_indices.copy(),
            "labeled_indices": labeled_indices.copy(),
            "initialization_strategy": initialization_strategy,
        }
)

        if round_id == n_rounds:
            break

        probabilities = model.predict_proba(
            X_pool[unlabeled_indices]
        )[:, 1]

        if strategy == "random":
            selected_indices = select_random(
                unlabeled_indices,
                batch_size=batch_size,
                random_state=random_state + round_id,
            )

        elif strategy == "greedy":
            selected_indices = select_greedy(
                unlabeled_indices,
                probabilities,
                batch_size=batch_size,
            )

        elif strategy == "uncertainty_topk":
            selected_indices = select_uncertainty_topk(
                unlabeled_indices,
                probabilities,
                batch_size=batch_size,
            )

        elif strategy == "uncertainty_diverse":
            selected_indices = select_uncertainty_diverse(
                X_pool=X_pool,
                unlabeled_indices=unlabeled_indices,
                probabilities=probabilities,
                batch_size=batch_size,
                candidate_pool_size=100,
                random_state=random_state + round_id,
            )

        elif strategy == "query_by_committee":
            committee = train_committee(
                X_pool[labeled_indices],
                y_pool[labeled_indices],
                random_state=random_state + round_id,
            )

            disagreement_scores = compute_disagreement_scores(
                committee,
                X_pool[unlabeled_indices],
            )

            selected_indices = select_query_by_committee(
                unlabeled_indices=unlabeled_indices,
                disagreement_scores=disagreement_scores,
                batch_size=batch_size,
            )

        else:
            raise ValueError(
                "strategy must be one of: 'random', 'greedy', "
                "'uncertainty_topk', 'uncertainty_diverse', "
                "'query_by_committee'"
            )

        history[-1]["selected_indices"] = selected_indices.copy()

        labeled_indices, unlabeled_indices = update_pool(
            labeled_indices,
            unlabeled_indices,
            selected_indices,
        )


        

    return pd.DataFrame(history)


def run_repeated_simulations(
    X,
    y,
    strategy,
    seeds,
    **kwargs,
):
    """
    Run several independent active-learning simulations.

    Parameters
    ----------
    seeds : iterable
        Random seeds used for independent simulations.

    kwargs
        Passed directly to run_simulation().

    Returns
    -------
    pd.DataFrame
        Concatenated history from all runs.
    """

    histories = []

    for seed in seeds:

        history = run_simulation(
            X,
            y,
            strategy=strategy,
            random_state=seed,
            **kwargs,
        )

        history["seed"] = seed

        histories.append(history)

    return pd.concat(
        histories,
        ignore_index=True,
    )