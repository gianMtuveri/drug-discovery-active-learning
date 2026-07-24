"""Generate predictive-performance figures from round_summary.csv."""

from src.regression.plotting import (
    METRICS,
    load_round_summary,
    plot_metric_panel,
    plot_metric_vs_round,
    set_plot_style,
)

from _cli import parse_single_benchmark_args


PREDICTION_METRICS = (
    "rmse",
    "mae",
    "r2",
    "pearson",
)


def main() -> None:
    args = parse_single_benchmark_args(
        "Plot regression prediction metrics."
    )

    set_plot_style()

    frame = load_round_summary(
        args.input_dir / "round_summary.csv"
    )

    print("Generating prediction figures...")

    for metric in PREDICTION_METRICS:
        plot_metric_vs_round(
            frame,
            metric,
            args.output_dir / METRICS[metric].filename,
        )

    plot_metric_panel(
        frame,
        PREDICTION_METRICS,
        args.output_dir / "prediction_metrics_panel.png",
    )


if __name__ == "__main__":
    main()