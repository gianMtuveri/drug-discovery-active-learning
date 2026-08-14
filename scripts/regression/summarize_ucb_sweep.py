import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGETS = [
    "EGFR",
    "JAK2",
    "PARP1",
    "BRAF",
    "ABL1",
    "SRC",
    "VEGFR2",
    "CDK2",
    "DRD2",
    "CA2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multi-target UCB regression sweeps and compare "
            "each beta value against beta=0."
        )
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Targets to include in the analysis.",
    )

    parser.add_argument(
        "--round",
        type=int,
        default=10,
        help="Active-learning round used for the final comparison.",
    )

    parser.add_argument(
        "--retention",
        type=float,
        default=0.98,
        help=(
            "Minimum fraction of beta=0 Top-20 discovery performance "
            "that must be retained by the constrained winner."
        ),
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
    )

    return parser.parse_args()


def load_target_summary(
    target: str,
    tables_dir: Path,
) -> pd.DataFrame:
    path = (
        tables_dir
        / f"{target.lower()}_regression_ucb_sweep_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing UCB summary for {target}: {path}"
        )

    df = pd.read_csv(path)
    df["target"] = target

    required_columns = {
        "beta",
        "round",
        "rmse_mean",
        "r2_mean",
        "pearson_mean",
        "best_discovered_mean",
        "top20_mean_discovered_mean",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return df


def build_target_comparison(
    summaries: pd.DataFrame,
    final_round: int,
    retention: float,
) -> pd.DataFrame:
    final = summaries[
        summaries["round"] == final_round
    ].copy()

    if final.empty:
        raise ValueError(
            f"No UCB results found for round {final_round}."
        )

    rows = []

    for target, target_df in final.groupby("target"):
        target_df = target_df.sort_values("beta").copy()

        baseline = target_df[
            np.isclose(target_df["beta"], 0.0)
        ]

        if len(baseline) != 1:
            raise ValueError(
                f"{target}: expected exactly one beta=0 row, "
                f"found {len(baseline)}."
            )

        baseline_row = baseline.iloc[0]

        baseline_rmse = float(
            baseline_row["rmse_mean"]
        )
        baseline_top20 = float(
            baseline_row["top20_mean_discovered_mean"]
        )

        for _, row in target_df.iterrows():
            beta = float(row["beta"])
            rmse = float(row["rmse_mean"])
            top20 = float(
                row["top20_mean_discovered_mean"]
            )

            if np.isclose(baseline_top20, 0.0):
                retention_ratio = np.nan
            else:
                retention_ratio = top20 / baseline_top20

            rows.append(
                {
                    "target": target,
                    "beta": beta,
                    "rmse": rmse,
                    "r2": float(row["r2_mean"]),
                    "pearson": float(
                        row["pearson_mean"]
                    ),
                    "best_discovered": float(
                        row["best_discovered_mean"]
                    ),
                    "top20": top20,
                    "delta_rmse_vs_beta0": (
                        rmse - baseline_rmse
                    ),
                    "delta_top20_vs_beta0": (
                        top20 - baseline_top20
                    ),
                    "top20_retention_ratio": (
                        retention_ratio
                    ),
                    "rmse_improved": (
                        rmse < baseline_rmse
                    ),
                    "top20_preserved": (
                        retention_ratio >= retention
                        if np.isfinite(retention_ratio)
                        else False
                    ),
                }
            )

    return pd.DataFrame(rows)


def mark_constrained_winners(
    target_comparison: pd.DataFrame,
) -> pd.DataFrame:
    result = target_comparison.copy()
    result["constrained_winner"] = False

    for target, target_df in result.groupby("target"):
        eligible = target_df[
            target_df["top20_preserved"]
        ]

        if eligible.empty:
            continue

        winner_index = eligible["rmse"].idxmin()
        result.loc[
            winner_index,
            "constrained_winner",
        ] = True

    return result


def build_beta_summary(
    target_comparison: pd.DataFrame,
) -> pd.DataFrame:
    n_targets = target_comparison["target"].nunique()

    summary = (
        target_comparison
        .groupby("beta")
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_pearson=("pearson", "mean"),
            mean_top20=("top20", "mean"),
            mean_delta_rmse=(
                "delta_rmse_vs_beta0",
                "mean",
            ),
            median_delta_rmse=(
                "delta_rmse_vs_beta0",
                "median",
            ),
            mean_delta_top20=(
                "delta_top20_vs_beta0",
                "mean",
            ),
            median_delta_top20=(
                "delta_top20_vs_beta0",
                "median",
            ),
            mean_top20_retention=(
                "top20_retention_ratio",
                "mean",
            ),
            rmse_improved_targets=(
                "rmse_improved",
                "sum",
            ),
            top20_preserved_targets=(
                "top20_preserved",
                "sum",
            ),
            constrained_wins=(
                "constrained_winner",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("beta")
    )

    summary["n_targets"] = n_targets

    summary["rmse_improved_fraction"] = (
        summary["rmse_improved_targets"]
        / n_targets
    )

    summary["top20_preserved_fraction"] = (
        summary["top20_preserved_targets"]
        / n_targets
    )

    return summary


def print_beta_summary(
    beta_summary: pd.DataFrame,
    retention: float,
):
    display_columns = [
        "beta",
        "mean_delta_rmse",
        "rmse_improved_targets",
        "mean_delta_top20",
        "mean_top20_retention",
        "top20_preserved_targets",
        "constrained_wins",
    ]

    print("\nCross-target UCB summary")
    print("------------------------")
    print(
        beta_summary[display_columns]
        .to_string(
            index=False,
            formatters={
                "beta": lambda value: f"{value:g}",
                "mean_delta_rmse": (
                    lambda value: f"{value:+.4f}"
                ),
                "mean_delta_top20": (
                    lambda value: f"{value:+.4f}"
                ),
                "mean_top20_retention": (
                    lambda value: f"{value:.4f}"
                ),
            },
        )
    )

    print(
        "\nA negative ΔRMSE indicates better prediction "
        "than beta=0."
    )

    print(
        "A positive ΔTop-20 indicates better lead discovery "
        "than beta=0."
    )

    print(
        f"Discovery-preservation threshold: "
        f"{retention:.1%}"
    )


def print_target_winners(
    target_comparison: pd.DataFrame,
):
    winners = (
        target_comparison[
            target_comparison["constrained_winner"]
        ][
            [
                "target",
                "beta",
                "rmse",
                "delta_rmse_vs_beta0",
                "top20",
                "top20_retention_ratio",
            ]
        ]
        .sort_values("target")
    )

    print("\nConstrained winner by target")
    print("----------------------------")

    if winners.empty:
        print(
            "No beta values satisfied the discovery-retention "
            "constraint."
        )
        return

    print(
        winners.to_string(
            index=False,
            formatters={
                "beta": lambda value: f"{value:g}",
                "rmse": lambda value: f"{value:.4f}",
                "delta_rmse_vs_beta0": (
                    lambda value: f"{value:+.4f}"
                ),
                "top20": lambda value: f"{value:.4f}",
                "top20_retention_ratio": (
                    lambda value: f"{value:.4f}"
                ),
            },
        )
    )


def main():
    args = parse_args()

    if not 0 < args.retention <= 1:
        raise ValueError(
            "--retention must be greater than 0 and at most 1."
        )

    summaries = pd.concat(
        [
            load_target_summary(
                target=target,
                tables_dir=args.tables_dir,
            )
            for target in args.targets
        ],
        ignore_index=True,
    )

    target_comparison = build_target_comparison(
        summaries=summaries,
        final_round=args.round,
        retention=args.retention,
    )

    target_comparison = mark_constrained_winners(
        target_comparison
    )

    beta_summary = build_beta_summary(
        target_comparison
    )

    output_prefix = (
        args.tables_dir
        / f"regression_ucb_multitarget_round{args.round}"
    )

    target_output = Path(
        f"{output_prefix}_target_comparison.csv"
    )

    beta_output = Path(
        f"{output_prefix}_beta_summary.csv"
    )

    target_comparison.to_csv(
        target_output,
        index=False,
    )

    beta_summary.to_csv(
        beta_output,
        index=False,
    )

    print_beta_summary(
        beta_summary=beta_summary,
        retention=args.retention,
    )

    print_target_winners(
        target_comparison=target_comparison,
    )

    print(f"\nSaved: {target_output}")
    print(f"Saved: {beta_output}")


if __name__ == "__main__":
    main()