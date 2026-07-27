"""Generate runtime and uncertainty figures from round_summary.csv."""

from src.regression.plotting import (
    METRICS,
    load_round_summary,
    plot_metric_vs_round,
    set_plot_style,
)

from _cli import parse_single_benchmark_args


def main() -> None:
    args = parse_single_benchmark_args("Plot runtime and uncertainty diagnostics.")
    set_plot_style()
    frame = load_round_summary(args.input_dir / "round_summary.csv")

    print("Generating runtime and uncertainty figures...")
    for metric in (
        "round_runtime_seconds",
        "pool_mean_uncertainty",
        "pool_max_uncertainty",
    ):
        plot_metric_vs_round(
            frame,
            metric,
            args.output_dir / METRICS[metric].filename,
        )


if __name__ == "__main__":
    main()
