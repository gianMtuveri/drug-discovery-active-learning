from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.targets import load_target_regression_evaluation
from src.regression.evaluation import bemis_murcko_scaffolds
from src.regression.simulation import (
    concatenate_simulation_histories,
    run_regression_simulation,
)

from src.regression.benchmarking import (
    SUMMARY_METRICS,
    V05_DISCOVERY_METRICS,
    save_benchmark_outputs,
)

DEFAULT_MODELS = [
    "random_forest",
    "extra_trees",
    "bayesian_ridge",
    "gaussian_process",
]

DEFAULT_V050_BENCHMARK_ROOT = Path(
    "results/regression/benchmarks/v050_egfr"
)


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
            "ucb_qed",
            "ucb_diversity",
            "ucb_qed_diversity",
            "uncertainty_diverse",
        ],
        default="ucb",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Exploration weight used by UCB.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for benchmark outputs. "
            "By default, outputs are isolated by strategy under the "
            "canonical v0.5 EGFR benchmark directory."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y, qed, affinity_nm, canonical_smiles = (
        load_target_regression_evaluation(args.target)
    )
    # Calculate scaffolds once per dataset, not once per campaign or round.
    scaffolds = bemis_murcko_scaffolds(canonical_smiles)

    campaign_histories: list[pd.DataFrame] = []

    for model_name in args.models:
        print(f"\nStrategy: {args.strategy}; model: {model_name}")

        for seed in range(args.seeds):
            print(
                f"  Seed {seed + 1}/{args.seeds}",
                end="\r",
            )

            history = run_regression_simulation(
                X=X,
                y=y,
                qed=qed,
                affinity_nm=affinity_nm,
                canonical_smiles=canonical_smiles,
                scaffolds=scaffolds,
                model_name=model_name,
                strategy=args.strategy,
                beta=args.beta,
                random_state=seed,
                n_initial=20,
                batch_size=10,
                n_rounds=args.rounds,
                test_size=0.2,
            ).copy()

            # Add explicitly even if simulation.py already records them.
            history["model"] = model_name
            history["target"] = args.target

            campaign_histories.append(history)

        print(
            f"  Completed {args.seeds} seeds."
        )

    results = concatenate_simulation_histories(campaign_histories)

    if args.output_dir is None:
        output_dir = DEFAULT_V050_BENCHMARK_ROOT / args.strategy
    else:
        output_dir = args.output_dir


    config = {
        "target": args.target,
        "models": args.models,
        "strategy": args.strategy,
        "beta": args.beta,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "n_initial": 20,
        "batch_size": 10,
        "test_size": 0.2,
        "activity_threshold_nm": 500.0,
        "qed_threshold": 0.6,
        "joint_discovery_auc_integration": "trapezoidal",
        "joint_discovery_auc_x_axis": "cumulative newly acquired measurements",
        "joint_diversity": "mean pairwise (1 - Tanimoto similarity)",
        "joint_diversity_fewer_than_two": "NaN",
    }


    paths = save_benchmark_outputs(
        history=results,
        output_dir=output_dir,
        config=config,
        reference_model=args.models[0],
        metrics=[*SUMMARY_METRICS, *V05_DISCOVERY_METRICS],
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
