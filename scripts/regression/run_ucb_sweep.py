import argparse
from pathlib import Path

import pandas as pd

from src.data.targets import load_target_regression
from src.regression.simulation import run_regression_simulation


DEFAULT_BETAS = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated regression active-learning simulations "
            "using Upper Confidence Bound acquisition."
        )
    )

    parser.add_argument(
        "--target",
        required=True,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--n-initial",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=DEFAULT_BETAS,
    )

    return parser.parse_args()


def beta_label(beta: float) -> str:
    return f"ucb_beta_{beta:g}"


def main():
    args = parse_args()

    if any(beta < 0 for beta in args.betas):
        raise ValueError("All beta values must be non-negative.")

    X, y = load_target_regression(args.target)

    histories = []

    for beta in args.betas:
        print(f"\nRunning beta={beta:g}")

        for seed in range(args.seeds):
            history = run_regression_simulation(
                X=X,
                y=y,
                strategy="ucb",
                beta=beta,
                random_state=seed,
                n_initial=args.n_initial,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                test_size=args.test_size,
            )

            history["strategy_label"] = beta_label(beta)

            histories.append(history)

    history = pd.concat(
        histories,
        ignore_index=True,
    )

    summary = (
        history
        .groupby(
            [
                "strategy",
                "strategy_label",
                "beta",
                "round",
            ],
            dropna=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            best_discovered_mean=(
                "best_discovered",
                "mean",
            ),
            best_discovered_std=(
                "best_discovered",
                "std",
            ),
            top20_mean_discovered_mean=(
                "top20_mean_discovered",
                "mean",
            ),
            top20_mean_discovered_std=(
                "top20_mean_discovered",
                "std",
            ),
            selected_mean_prediction_mean=(
                "selected_mean_prediction",
                "mean",
            ),
            selected_mean_uncertainty_mean=(
                "selected_mean_uncertainty",
                "mean",
            ),
            selected_mean_true_affinity_mean=(
                "selected_mean_true_affinity",
                "mean",
            ),
            selected_best_true_affinity_mean=(
                "selected_best_true_affinity",
                "mean",
            ),
        )
        .reset_index()
    )

    output_dir = Path("results/tables")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        output_dir
        / f"{args.target.lower()}_regression_ucb_sweep"
    )

    history_path = Path(f"{prefix}_history.csv")
    summary_path = Path(f"{prefix}_summary.csv")

    history.to_csv(
        history_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    final_summary = (
        summary[
            summary["round"] == args.rounds
        ]
        .sort_values("beta")
    )

    columns = [
        "beta",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
        "pearson_mean",
        "best_discovered_mean",
        "top20_mean_discovered_mean",
    ]

    print("\nFinal-round UCB results")
    print(
        final_summary[columns].to_string(
            index=False,
        )
    )

    print(f"\nSaved history: {history_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()