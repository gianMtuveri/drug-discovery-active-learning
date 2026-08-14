from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold

from src.data.targets import load_target_regression
from src.regression.benchmarking import (
    save_supervised_benchmark_outputs,
)
from src.regression.models import (
    make_regression_surrogate,
)


DEFAULT_MODELS = [
    "random_forest",
    "extra_trees",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate supervised regression performance using "
            "repeated random K-fold cross-validation."
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
        help="Regression surrogate models to compare.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeated cross-validation runs.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed controlling cross-validation splits.",
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


def safe_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Pearson correlation safely.

    Returns NaN when either vector has zero variance.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.size < 2:
        return float("nan")

    if np.isclose(
        np.std(y_true),
        0.0,
    ):
        return float("nan")

    if np.isclose(
        np.std(y_pred),
        0.0,
    ):
        return float("nan")

    return float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate supervised regression metrics.
    """
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "pearson": safe_pearson(
            y_true,
            y_pred,
        ),
    }


def main() -> None:
    args = parse_args()

    if args.folds < 2:
        raise ValueError(
            "--folds must be at least 2."
        )

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )

    X, y = load_target_regression(
        args.target
    )

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError(
            "X and y contain different numbers of samples."
        )

    if len(y) < args.folds:
        raise ValueError(
            "The number of folds cannot exceed the "
            "number of samples."
        )

    splitter = RepeatedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.random_state,
    )

    splits = list(
        splitter.split(X)
    )

    expected_splits = (
        args.folds * args.repeats
    )

    if len(splits) != expected_splits:
        raise RuntimeError(
            "Unexpected number of cross-validation splits: "
            f"expected {expected_splits}, got {len(splits)}."
        )

    rows: list[dict[str, object]] = []

    for model_name in args.models:
        print(
            f"\nModel: {model_name}"
        )

        for split_index, (
            train_indices,
            test_indices,
        ) in enumerate(splits):
            repeat = (
                split_index // args.folds
            )

            fold = (
                split_index % args.folds
            )

            # Give each fitted estimator a reproducible but distinct
            # random seed.
            model_random_state = (
                args.random_state
                + split_index
            )

            model = make_regression_surrogate(
                model_name=model_name,
                random_state=model_random_state,
            )

            model.fit(
                X[train_indices],
                y[train_indices],
            )

            predictions = model.predict(
                X[test_indices]
            )

            metrics = evaluate_predictions(
                y_true=y[test_indices],
                y_pred=predictions,
            )

            rows.append(
                {
                    "target": args.target,
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "split": split_index,
                    "random_state": (
                        model_random_state
                    ),
                    "n_train": len(
                        train_indices
                    ),
                    "n_test": len(
                        test_indices
                    ),
                    **metrics,
                }
            )

            print(
                "  Split "
                f"{split_index + 1}/"
                f"{expected_splits}",
                end="\r",
            )

        print(
            "  Completed "
            f"{expected_splits} evaluations."
        )

    history = pd.DataFrame(
        rows
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = Path(
            "results/regression/"
            "supervised_benchmarks"
        ) / (
            f"{args.target.lower()}_"
            f"random_cv_"
            f"{timestamp}"
        )
    else:
        output_dir = args.output_dir

    config = {
        "target": args.target,
        "models": args.models,
        "validation": "repeated_random_kfold",
        "folds": args.folds,
        "repeats": args.repeats,
        "random_state": args.random_state,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }

    paths = save_supervised_benchmark_outputs(
        history=history,
        output_dir=output_dir,
        config=config,
    )

    print(
        f"\nSaved {len(history)} evaluations to:"
        f"\n{output_dir}"
    )

    print("\nGenerated files:")

    for name, path in paths.items():
        print(
            f"  {name:12s} {path}"
        )

    print("\nEvaluations per model:")

    print(
        history.groupby("model")
        .size()
        .to_string()
    )

    print("\nSupervised CV summary:")

    summary = pd.read_csv(
        paths["summary"]
    )

    display_columns = [
        "model",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "pearson_mean",
        "pearson_std",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    main()