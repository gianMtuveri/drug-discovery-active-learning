import pandas as pd

from src.toy_data import make_toy_dataset
from src.simulation import run_repeated_simulations


def summarize_results(history):
    summary = (
        history
        .groupby(["initialization_strategy", "strategy", "round"])
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            actives_mean=("n_actives_found", "mean"),
            actives_std=("n_actives_found", "std"),
        )
        .reset_index()
    )

    return summary


def main():
    X, y = make_toy_dataset(n_samples=1000, random_state=42)

    all_histories = []

    for initialization_strategy in ["random", "diverse"]:
        for selection_strategy in ["random", "greedy", "uncertainty_topk", "uncertainty_diverse", "query_by_committee"]:
            history = run_repeated_simulations(
                X,
                y,
                strategy=selection_strategy,
                initialization_strategy=initialization_strategy,
                seeds=range(50),
                n_initial=20,
                batch_size=10,
                n_rounds=10,
            )

            all_histories.append(history)

    history = pd.concat(all_histories, ignore_index=True)
    summary = summarize_results(history)

    print(summary)

    history.to_csv(
        "results/tables/toy_initialization_comparison_history.csv",
        index=False,
    )

    summary.to_csv(
        "results/tables/toy_initialization_comparison_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()