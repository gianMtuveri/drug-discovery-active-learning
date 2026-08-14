"""Generate molecular-discovery figures from round_summary.csv."""

from src.regression.plotting import (
    METRICS,
    load_round_summary,
    plot_metric_vs_round,
    set_plot_style,
)

from _cli import parse_single_benchmark_args


def main() -> None:
    args = parse_single_benchmark_args("Plot regression discovery metrics.")
    set_plot_style()
    frame = load_round_summary(args.input_dir / "round_summary.csv")

    print("Generating discovery figures...")
    for metric in (
        "best_discovered",
        "top20_mean_discovered",
        "mean_discovered",
        "fraction_best_found",
        "distance_to_best",
    ):
        plot_metric_vs_round(
            frame,
            metric,
            args.output_dir / METRICS[metric].filename,
        )


if __name__ == "__main__":
    main()
