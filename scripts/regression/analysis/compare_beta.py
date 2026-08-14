"""Compare two aggregated final-round regression benchmark summaries."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


METRICS = (
    "rmse",
    "mae",
    "r2",
    "pearson",
    "best_discovered",
    "top20_mean_discovered",
    "mean_discovered",
)

HIGHER_IS_BETTER = {
    "rmse": False,
    "mae": False,
    "r2": True,
    "pearson": True,
    "best_discovered": True,
    "top20_mean_discovered": True,
    "mean_discovered": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two aggregated final-round benchmark summaries."
    )
    parser.add_argument("--beta1", type=Path, required=True)
    parser.add_argument("--beta2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    frame = pd.read_csv(path)

    required = {"model"}
    for metric in METRICS:
        required.update({f"{metric}_mean", f"{metric}_std"})

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if frame["model"].duplicated().any():
        duplicated = sorted(frame.loc[frame["model"].duplicated(), "model"].unique())
        raise ValueError(f"{path} contains duplicated model rows: {duplicated}")

    columns = ["model"]
    if "n_seeds" in frame.columns:
        columns.append("n_seeds")

    for metric in METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std"))

    return frame[columns].rename(
        columns={
            column: f"{column}_{suffix}"
            for column in columns
            if column != "model"
        }
    )


def determine_winner(
    delta: float,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> str:
    if np.isclose(delta, 0.0, atol=tolerance, rtol=0.0):
        return "tie"

    beta2_is_better = delta > 0 if higher_is_better else delta < 0
    return "beta2" if beta2_is_better else "beta1"


def relative_change_percent(beta1: float, beta2: float) -> float:
    """Signed percentage change from beta 1 to beta 2."""
    if np.isclose(beta1, 0.0):
        return np.nan
    return 100.0 * (beta2 - beta1) / abs(beta1)


def improvement_percent(
    beta1: float,
    beta2: float,
    higher_is_better: bool,
) -> float:
    """Percentage change oriented so positive values always favor beta 2."""
    change = relative_change_percent(beta1, beta2)
    if np.isnan(change):
        return np.nan
    return change if higher_is_better else -change


def validate_model_sets(beta1: pd.DataFrame, beta2: pd.DataFrame) -> None:
    models1 = set(beta1["model"])
    models2 = set(beta2["model"])

    if models1 == models2:
        return

    only_beta1 = sorted(models1 - models2)
    only_beta2 = sorted(models2 - models1)

    details = []
    if only_beta1:
        details.append(f"only in beta1: {only_beta1}")
    if only_beta2:
        details.append(f"only in beta2: {only_beta2}")

    raise ValueError(
        "The two summaries contain different model sets; " + "; ".join(details)
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    beta1 = load_summary(args.beta1, "beta1")
    beta2 = load_summary(args.beta2, "beta2")
    validate_model_sets(beta1, beta2)

    wide = (
        beta1.merge(beta2, on="model", how="inner", validate="one_to_one")
        .sort_values("model")
        .reset_index(drop=True)
    )

    long_rows: list[dict[str, object]] = []

    for _, row in wide.iterrows():
        model = row["model"]

        for metric in METRICS:
            beta1_mean = float(row[f"{metric}_mean_beta1"])
            beta2_mean = float(row[f"{metric}_mean_beta2"])
            delta = beta2_mean - beta1_mean
            higher_is_better = HIGHER_IS_BETTER[metric]

            long_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "beta1_mean": beta1_mean,
                    "beta1_std": float(row[f"{metric}_std_beta1"]),
                    "beta2_mean": beta2_mean,
                    "beta2_std": float(row[f"{metric}_std_beta2"]),
                    "delta_beta2_minus_beta1": delta,
                    "relative_change_percent": relative_change_percent(
                        beta1_mean, beta2_mean
                    ),
                    "improvement_percent": improvement_percent(
                        beta1_mean, beta2_mean, higher_is_better
                    ),
                    "winner": determine_winner(delta, higher_is_better),
                }
            )

    long_comparison = pd.DataFrame(long_rows)

    winner_summary = (
        long_comparison.groupby(["metric", "winner"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for column in ("beta1", "beta2", "tie"):
        if column not in winner_summary.columns:
            winner_summary[column] = 0

    winner_summary = winner_summary[["metric", "beta1", "beta2", "tie"]]

    wide.to_csv(args.output / "beta_comparison_wide.csv", index=False)
    long_comparison.to_csv(args.output / "beta_comparison_long.csv", index=False)
    winner_summary.to_csv(args.output / "beta_comparison_winners.csv", index=False)

    display_columns = [
        "model",
        "metric",
        "beta1_mean",
        "beta2_mean",
        "delta_beta2_minus_beta1",
        "relative_change_percent",
        "improvement_percent",
        "winner",
    ]

    print("\nFinal-round beta comparison\n")
    print(long_comparison[display_columns].round(4).to_string(index=False))

    print("\nWinner counts\n")
    print(winner_summary.to_string(index=False))

    print(f"\nSaved outputs to: {args.output}")


if __name__ == "__main__":
    main()