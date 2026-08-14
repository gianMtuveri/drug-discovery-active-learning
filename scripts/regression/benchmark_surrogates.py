from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.targets import load_target_regression
from src.regression.simulation import run_regression_simulation

from datetime import datetime
from pathlib import Path

from src.regression.benchmarking import (
    save_benchmark_outputs,
)

DEFAULT_MODELS = [
    "random_forest",
    "extra_trees",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare regression surrogate models using identical "
            "active-learning campaigns."
        )
    )

    parser.add_argument(
        "--target",
        default="EGFR",
        help="Regression target to load.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Surrogate models to compare.",
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "random",
            "greedy",
            "uncertainty",
            "ucb",
        ],
        default="ucb",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Exploration weight used by UCB.",
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
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for benchmark outputs. "
            "A timestamped directory is used by default."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y = load_target_regression(args.target)

    campaign_histories: list[pd.DataFrame] = []

    for model_name in args.models:
        print(f"\nModel: {model_name}")

        for seed in range(args.seeds):
            print(
                f"  Seed {seed + 1}/{args.seeds}",
                end="\r",
            )

            history = run_regression_simulation(
                X=X,
                y=y,
                model_name=model_name,
                strategy=args.strategy,
                beta=args.beta,
                random_state=seed,
                n_rounds=args.rounds,
            ).copy()

            # Add explicitly even if simulation.py already records them.
            history["model"] = model_name
            history["target"] = args.target

            campaign_histories.append(history)

        print(
            f"  Completed {args.seeds} seeds."
        )

    results = pd.concat(
        campaign_histories,
        ignore_index=True,
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = Path(
            "results/regression/benchmarks"
        ) / (
            f"{args.target.lower()}_"
            f"{args.strategy}_"
            f"{timestamp}"
        )
    else:
        output_dir = args.output_dir


    config = {
        "target": args.target,
        "models": args.models,
        "strategy": args.strategy,
        "beta": args.beta,
        "seeds": args.seeds,
        "rounds": args.rounds,
    }


    paths = save_benchmark_outputs(
        history=results,
        output_dir=output_dir,
        config=config,
        reference_model=args.models[0],
    )


    print(
        f"\nSaved {len(results)} history rows to:"
        f"\n{output_dir}"
    )

    print("\nGenerated files:")

    for name, path in paths.items():
        print(f"  {name:28s} {path}")

    print("\nRows per model:")
    print(
        results.groupby("model")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()