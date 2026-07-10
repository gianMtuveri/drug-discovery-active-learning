import argparse
from pathlib import Path

import pandas as pd

from src.data.targets import load_target_regression
from src.regression.simulation import run_regression_simulation


STRATEGIES = [
    "random",
    "greedy",
    "uncertainty",
    "uncertainty_diverse",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_target_regression(args.target)

    histories = []

    for strategy in STRATEGIES:
        for seed in range(args.seeds):
            history = run_regression_simulation(
                X=X,
                y=y,
                strategy=strategy,
                random_state=seed,
                n_initial=args.n_initial,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                candidate_pool_size=args.candidate_pool_size,
            )
            histories.append(history)

    history = pd.concat(histories, ignore_index=True)

    summary = (
        history
        .groupby(["strategy", "round"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            best_discovered_mean=("best_discovered", "mean"),
            best_discovered_std=("best_discovered", "std"),
            top20_mean_discovered_mean=("top20_mean_discovered", "mean"),
            top20_mean_discovered_std=("top20_mean_discovered", "std"),
        )
        .reset_index()
    )

    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_dir / f"{args.target.lower()}_regression_active_learning"

    history.to_csv(f"{prefix}_history.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)

    print(
        summary[summary["round"] == args.rounds]
        .sort_values("rmse_mean")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()