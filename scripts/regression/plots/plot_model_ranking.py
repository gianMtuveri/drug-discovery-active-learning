"""Generate final-round model rankings."""

from src.regression.plotting import (
    load_final_round_summary,
    plot_final_metric_ranking,
    set_plot_style,
)

from _cli import parse_single_benchmark_args


def main() -> None:
    args = parse_single_benchmark_args("Plot final-round model rankings.")
    set_plot_style()
    frame = load_final_round_summary(args.input_dir / "final_round_summary.csv")

    print("Generating final-round rankings...")
    for metric in ("rmse", "r2", "best_discovered", "top20_mean_discovered"):
        plot_final_metric_ranking(
            frame,
            metric,
            args.output_dir / f"final_{metric}_ranking.png",
        )


if __name__ == "__main__":
    main()
