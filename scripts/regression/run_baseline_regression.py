import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.targets import load_target_regression
from src.regression.models import make_regression_model
from src.regression.evaluation import (
    evaluate_regression,
    best_affinity,
    top_k_mean_affinity,
    top_k_enrichment,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "bayesian_ridge"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_target_regression(args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = make_regression_model(
        model_name=args.model,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "residual": y_pred - y_test,
        }
    )

    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"{args.target.lower()}_{args.model}_regression_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    print(f"Saved predictions: {predictions_path}")

    metrics = evaluate_regression(y_test, y_pred)

    selected_metrics = {
        "best_affinity_test": best_affinity(y_test),
        "top20_mean_affinity_test": top_k_mean_affinity(y_test, k=20),
        "top100_enrichment_test": top_k_enrichment(y_test, y, k=100),
    }

    row = {
        "target": args.target,
        "model": args.model,
        "train_molecules": len(y_train),
        "test_molecules": len(y_test),
        **metrics,
        **selected_metrics,
    }

    output_path = output_dir / f"{args.target.lower()}_{args.model}_regression_baseline.csv"

    pd.DataFrame([row]).to_csv(output_path, index=False)

    print(pd.DataFrame([row]).T)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()