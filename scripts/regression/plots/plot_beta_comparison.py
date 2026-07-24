"""Compare round-level benchmark metrics between two exploration weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.regression.plotting import (
    load_round_summary,
    plot_beta_comparison,
    set_plot_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two UCB beta benchmarks.")
    parser.add_argument("--beta1-dir", type=Path, required=True)
    parser.add_argument("--beta2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Show standard-deviation bands. Disabled by default to reduce visual clutter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_plot_style()

    beta1 = load_round_summary(args.beta1_dir / "round_summary.csv")
    beta2 = load_round_summary(args.beta2_dir / "round_summary.csv")

    beta1_value = float(beta1["beta"].dropna().iloc[0])
    beta2_value = float(beta2["beta"].dropna().iloc[0])
    frames = {beta1_value: beta1, beta2_value: beta2}

    print("Generating beta-comparison figures...")
    for metric in (
        "rmse",
        "r2",
        "best_discovered",
        "top20_mean_discovered",
        "fraction_best_found",
        "pool_mean_uncertainty",
    ):
        plot_beta_comparison(
            frames,
            metric,
            args.output_dir / f"{metric}_beta_comparison.png",
            show_std=args.show_std,
        )


if __name__ == "__main__":
    main()
