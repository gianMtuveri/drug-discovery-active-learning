import argparse

import numpy as np
import pandas as pd

from src.data.targets import load_target_regression
from src.regression.simulation import run_regression_simulation


METRICS = [
    "rmse",
    "mae",
    "r2",
    "pearson",
    "best_discovered",
    "top20_mean_discovered",
    "mean_discovered",
    "selected_mean_prediction",
    "selected_mean_uncertainty",
    "selected_mean_true_affinity",
    "selected_best_true_affinity",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default="EGFR",
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="CSV created before modular-engine integration.",
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "greedy",
            "uncertainty",
            "ucb",
        ],
        required=True,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_target_regression(
        args.target
    )

    generated = []

    for seed in range(args.seeds):
        history = run_regression_simulation(
            X=X,
            y=y,
            strategy=args.strategy,
            beta=args.beta,
            random_state=seed,
            n_rounds=args.rounds,
        )

        generated.append(history)

    new = pd.concat(
        generated,
        ignore_index=True,
    ).sort_values(
        ["seed", "round"]
    ).reset_index(drop=True)

    old = pd.read_csv(
        args.reference
    )

    old = old[
        (old["strategy"] == args.strategy)
        & (old["seed"] < args.seeds)
        & (old["round"] <= args.rounds)
    ]

    if args.strategy == "ucb":
        old = old[
            np.isclose(
                old["beta"],
                args.beta,
            )
        ]

    old = old.sort_values(
        ["seed", "round"]
    ).reset_index(drop=True)

    print(
        "Same number of rows:",
        len(old) == len(new),
    )

    if len(old) != len(new):
        raise AssertionError(
            "History row counts differ."
        )

    for metric in METRICS:
        if metric not in old.columns:
            print(
                f"{metric:40s} skipped "
                "(missing in reference)"
            )
            continue

        equal = np.allclose(
            old[metric].to_numpy(),
            new[metric].to_numpy(),
            equal_nan=True,
        )

        differences = np.abs(
            old[metric].to_numpy()
            - new[metric].to_numpy()
        )

        max_difference = (
            np.nanmax(differences)
            if not np.isnan(differences).all()
            else 0.0
        )

        print(
            f"{metric:40s} "
            f"equal={equal} "
            f"max_diff={max_difference:.12g}"
        )

        if not equal:
            raise AssertionError(
                f"Metric differs: {metric}"
            )

    print(
        "\nFull simulation equivalence passed."
    )


if __name__ == "__main__":
    main()