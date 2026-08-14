import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_metrics(target, model):
    path = Path(f"results/tables/{target.lower()}_{model}_regression_baseline.csv")
    return pd.read_csv(path).iloc[0]


def load_predictions(target, model):
    path = Path(f"results/tables/{target.lower()}_{model}_regression_predictions.csv")
    return pd.read_csv(path)


def plot_regression_report(target, model):
    metrics = load_metrics(target, model)
    pred = load_predictions(target, model)

    y_true = pred["y_true"].to_numpy()
    y_pred = pred["y_pred"].to_numpy()
    residual = pred["residual"].to_numpy()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(15, 12),
    )

    # 1. Predicted vs experimental
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, s=12, alpha=0.45)

    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)

    ax.set_xlabel("Experimental pAffinity")
    ax.set_ylabel("Predicted pAffinity")
    ax.set_title("Predicted vs experimental")
    ax.grid(alpha=0.25)

    # 2. Residuals vs experimental
    ax = axes[0, 1]
    ax.scatter(y_true, residual, s=12, alpha=0.45)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Experimental pAffinity")
    ax.set_ylabel("Residual (predicted - true)")
    ax.set_title("Residuals")
    ax.grid(alpha=0.25)

    # 3. Error histogram
    ax = axes[0, 2]
    ax.hist(residual, bins=50, alpha=0.8)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.grid(alpha=0.25)

    # 4. True vs predicted distributions
    ax = axes[1, 0]
    ax.hist(y_true, bins=50, alpha=0.55, label="Experimental")
    ax.hist(y_pred, bins=50, alpha=0.55, label="Predicted")
    ax.set_xlabel("pAffinity")
    ax.set_ylabel("Count")
    ax.set_title("Affinity distributions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 5. Absolute error vs experimental
    ax = axes[1, 1]
    abs_error = np.abs(residual)
    ax.scatter(y_true, abs_error, s=12, alpha=0.45)
    ax.set_xlabel("Experimental pAffinity")
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute error by potency")
    ax.grid(alpha=0.25)

    # 6. Q-Q plot of residuals
    ax = axes[1, 2]

    (osm, osr), (slope, intercept, r) = stats.probplot(
        residual,
        dist="norm",
    )

    ax.scatter(
        osm,
        osr,
        s=18,
        alpha=0.8,
        label="Residuals",
    )

    x = np.linspace(osm.min(), osm.max(), 100)

    ax.plot(
        x,
        slope * x + intercept,
        color="red",
        linewidth=2,
        label="Normal reference",
    )

    ax.set_title("Residual normality (Q-Q)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered residuals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 7. Calibration curve by prediction decile
    ax = axes[2, 0]

    calibration = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    calibration["bin"] = pd.qcut(
        calibration["y_pred"],
        q=10,
        duplicates="drop",
    )

    calib_summary = (
        calibration
        .groupby("bin", observed=True)
        .agg(
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
        )
    )

    ax.plot(
        calib_summary["mean_pred"],
        calib_summary["mean_true"],
        marker="o",
        label="Prediction deciles",
    )

    low = min(
        calib_summary["mean_pred"].min(),
        calib_summary["mean_true"].min(),
    )
    high = max(
        calib_summary["mean_pred"].max(),
        calib_summary["mean_true"].max(),
    )

    ax.plot([low, high], [low, high], linestyle="--", linewidth=1, label="Ideal (y = x)",)

    ax.set_xlabel("Mean predicted pAffinity")
    ax.set_ylabel("Mean experimental pAffinity")
    ax.set_title("Calibration by prediction decile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 8. Ranking quality: top predicted fractions
    ax = axes[2, 1]

    ranking = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).sort_values("y_pred", ascending=False)

    fractions = [0.01, 0.02, 0.05, 0.10, 0.20]
    mean_true_values = []

    ideal = (
        pd.DataFrame({"y_true": y_true})
        .sort_values("y_true", ascending=False)
    )

    ideal_values = []

    for frac in fractions:
        n = max(1, int(len(ranking) * frac))
        mean_true_values.append(ranking.head(n)["y_true"].mean())
        ideal_values.append(ideal.head(n)["y_true"].mean())

    x = [f * 100 for f in fractions]

    ax.plot(
        x,
        mean_true_values,
        marker="o",
        label="Model ranking",
    )

    ax.plot(
        x,
        ideal_values,
        marker="s",
        linestyle="--",
        label="Ideal ranking",
    )

    ax.axhline(
        y_true.mean(),
        linestyle="--",
        linewidth=1,
        label="Test-set mean",
    )

    ax.set_xlabel("Top predicted fraction (%)")
    ax.set_ylabel("Mean experimental pAffinity")
    ax.set_title("Ranking quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 9. Metrics summary
    ax = axes[2, 2]
    ax.axis("off")

    text = (
        "Regression report\n\n"

        "Dataset\n"
        f"Target: {target}\n"
        "Prediction target: pAffinity\n"
        "Features: Morgan FP (2048 bits)\n\n"

        "Data\n"
        f"Train molecules: {int(metrics['train_molecules'])}\n"
        f"Test molecules: {int(metrics['test_molecules'])}\n\n"

        "Performance\n"
        f"RMSE: {metrics['rmse']:.3f}\n"
        f"MAE: {metrics['mae']:.3f}\n"
        f"R²: {metrics['r2']:.3f}\n"
        f"Pearson r: {metrics['pearson']:.3f}\n\n"

        "Ranking\n"
        f"Best test pAffinity: {metrics['best_affinity_test']:.3f}\n"
        f"Top-20 mean: {metrics['top20_mean_affinity_test']:.3f}\n"
        f"Top-100 enrichment: {metrics['top100_enrichment_test']:.3f}"
    )

    ax.text(
        0,
        1,
        text,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.suptitle(
        f"{target} regression baseline: {model}",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{target.lower()}_{model}_regression_report.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print("Saved:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "bayesian_ridge"],
    )
    args = parser.parse_args()

    plot_regression_report(args.target, args.model)


if __name__ == "__main__":
    main()