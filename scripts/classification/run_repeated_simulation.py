import argparse
import pandas as pd

from src.targets import load_target_classification
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target name, e.g. EGFR",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of repeated simulations.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_target_classification(args.target)

    all_histories = []

    strategies = [
        "random",
        "greedy",
        "uncertainty_topk",
        "uncertainty_diverse",
        "query_by_committee",
    ]

    for initialization_strategy in ["random", "diverse"]:
        for selection_strategy in strategies:
            history = run_repeated_simulations(
                X,
                y,
                strategy=selection_strategy,
                initialization_strategy=initialization_strategy,
                seeds=range(args.seeds),
                n_initial=20,
                batch_size=10,
                n_rounds=10,
                test_size=0.2,
            )

            all_histories.append(history)

    history = pd.concat(all_histories, ignore_index=True)
    summary = summarize_results(history)

    output_prefix = f"results/tables/{args.target.lower()}_classification"

    history.to_csv(
        f"{output_prefix}_history.csv",
        index=False,
    )

    summary.to_csv(
        f"{output_prefix}_summary.csv",
        index=False,
    )

    print(
        summary[summary["round"] == 10]
        .sort_values(["initialization_strategy", "strategy"])
    )


if __name__ == "__main__":
    main()