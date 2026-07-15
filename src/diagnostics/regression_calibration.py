import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.diagnostics.results import RegressionUncertaintyResult


def _validate_regression_inputs(
    y_true: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate regression predictions and uncertainty estimates."""
    y_true = np.asarray(y_true, dtype=float)
    predicted_mean = np.asarray(predicted_mean, dtype=float)
    predicted_uncertainty = np.asarray(
        predicted_uncertainty,
        dtype=float,
    )

    arrays = {
        "y_true": y_true,
        "predicted_mean": predicted_mean,
        "predicted_uncertainty": predicted_uncertainty,
    }

    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

    if not (
        len(y_true)
        == len(predicted_mean)
        == len(predicted_uncertainty)
    ):
        raise ValueError(
            "y_true, predicted_mean, and predicted_uncertainty "
            "must have the same length."
        )

    if len(y_true) < 2:
        raise ValueError(
            "Regression uncertainty evaluation requires "
            "at least two samples."
        )

    if np.any(predicted_uncertainty < 0.0):
        raise ValueError(
            "predicted_uncertainty cannot contain "
            "negative values."
        )

    return (
        y_true,
        predicted_mean,
        predicted_uncertainty,
    )


def _safe_correlations(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate uncertainty-error correlations.

    Correlations are undefined when uncertainty is constant.
    In that case, return NaN rather than raising an exception.
    """
    if np.isclose(
        np.max(uncertainty),
        np.min(uncertainty),
    ):
        return np.nan, np.nan

    pearson = float(
        pearsonr(
            uncertainty,
            absolute_error,
        ).statistic
    )

    spearman = float(
        spearmanr(
            uncertainty,
            absolute_error,
        ).statistic
    )

    return pearson, spearman


def _equal_frequency_edges(
    uncertainty: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Build approximately equal-frequency uncertainty bins.

    Duplicate quantile boundaries are removed. Consequently,
    repeated or constant uncertainty values may produce fewer
    effective bins than requested.
    """
    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.unique(
        np.quantile(
            uncertainty,
            quantiles,
        )
    )

    if len(edges) < 2:
        value = float(uncertainty[0])
        epsilon = max(
            abs(value) * 1e-12,
            1e-12,
        )

        edges = np.array(
            [
                value - epsilon,
                value + epsilon,
            ],
            dtype=float,
        )

    actual_bins = len(edges) - 1

    if actual_bins < n_bins:
        warnings.warn(
            (
                "Equal-frequency uncertainty binning produced "
                f"{actual_bins} bin(s) instead of the requested "
                f"{n_bins} because uncertainty estimates contain "
                "duplicate values."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    return edges


def calculate_uncertainty_bins(
    y_true: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_uncertainty: np.ndarray,
    *,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Summarize prediction errors across uncertainty quantiles.

    The first bin contains the least-uncertain predictions and
    the final bin contains the most-uncertain predictions.
    """
    (
        y_true,
        predicted_mean,
        predicted_uncertainty,
    ) = _validate_regression_inputs(
        y_true,
        predicted_mean,
        predicted_uncertainty,
    )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    absolute_error = np.abs(
        y_true - predicted_mean
    )

    squared_error = (
        y_true - predicted_mean
    ) ** 2

    edges = _equal_frequency_edges(
        predicted_uncertainty,
        n_bins,
    )

    bin_indices = np.digitize(
        predicted_uncertainty,
        edges[1:-1],
        right=True,
    )

    rows: list[dict[str, float | int]] = []

    for bin_index in range(
        len(edges) - 1
    ):
        mask = bin_indices == bin_index
        count = int(mask.sum())

        if count == 0:
            continue

        rows.append(
            {
                "bin_index": bin_index,
                "lower_uncertainty": float(
                    edges[bin_index]
                ),
                "upper_uncertainty": float(
                    edges[bin_index + 1]
                ),
                "count": count,
                "fraction": (
                    count / len(y_true)
                ),
                "mean_uncertainty": float(
                    np.mean(
                        predicted_uncertainty[mask]
                    )
                ),
                "median_uncertainty": float(
                    np.median(
                        predicted_uncertainty[mask]
                    )
                ),
                "mae": float(
                    np.mean(
                        absolute_error[mask]
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            squared_error[mask]
                        )
                    )
                ),
                "median_absolute_error": float(
                    np.median(
                        absolute_error[mask]
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_empirical_interval_statistics(
    y_true: np.ndarray,
    tree_predictions: np.ndarray,
    *,
    interval_levels: tuple[float, ...] = (
        0.50,
        0.80,
        0.90,
        0.95,
    ),
) -> pd.DataFrame:
    """
    Evaluate empirical prediction intervals from ensemble predictions.

    For a nominal interval level L, the interval is calculated from
    the central ensemble quantiles:

        lower quantile = (1 - L) / 2
        upper quantile = 1 - (1 - L) / 2

    Example for an 80% interval:

        lower = 10th percentile
        upper = 90th percentile
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    tree_predictions = (
        _validate_tree_predictions(
            tree_predictions,
            n_samples=len(y_true),
        )
    )

    if not interval_levels:
        raise ValueError(
            "At least one interval level is required."
        )

    rows: list[dict[str, float]] = []

    for nominal_coverage in interval_levels:
        if not 0.0 < nominal_coverage < 1.0:
            raise ValueError(
                "Every interval level must be between 0 and 1."
            )

        tail_probability = (
            1.0 - nominal_coverage
        ) / 2.0

        lower_quantile = tail_probability
        upper_quantile = (
            1.0 - tail_probability
        )

        lower_bounds = np.quantile(
            tree_predictions,
            lower_quantile,
            axis=1,
        )

        upper_bounds = np.quantile(
            tree_predictions,
            upper_quantile,
            axis=1,
        )

        covered = (
            (y_true >= lower_bounds)
            & (y_true <= upper_bounds)
        )

        widths = (
            upper_bounds - lower_bounds
        )

        observed_coverage = float(
            np.mean(covered)
        )

        rows.append(
            {
                "nominal_coverage": float(
                    nominal_coverage
                ),
                "lower_quantile": float(
                    lower_quantile
                ),
                "upper_quantile": float(
                    upper_quantile
                ),
                "observed_coverage": (
                    observed_coverage
                ),
                "coverage_gap": float(
                    observed_coverage
                    - nominal_coverage
                ),
                "absolute_coverage_gap": float(
                    abs(
                        observed_coverage
                        - nominal_coverage
                    )
                ),
                "mean_interval_width": float(
                    np.mean(widths)
                ),
                "median_interval_width": float(
                    np.median(widths)
                ),
                "minimum_interval_width": float(
                    np.min(widths)
                ),
                "maximum_interval_width": float(
                    np.max(widths)
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_regression_uncertainty(
    y_true: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_uncertainty: np.ndarray,
    *,
    tree_predictions: np.ndarray | None = None,
    n_bins: int = 4,
    interval_levels: tuple[float, ...] = (
        0.50,
        0.80,
        0.90,
        0.95,
    ),
) -> RegressionUncertaintyResult:
    """
    Evaluate whether predicted uncertainty identifies larger errors.

    Pearson correlation measures approximately linear association
    between uncertainty and absolute error.

    Spearman correlation measures whether molecules ranked as more
    uncertain also tend to have larger errors, making it especially
    relevant for active-learning acquisition.
    """
    (
        y_true,
        predicted_mean,
        predicted_uncertainty,
    ) = _validate_regression_inputs(
        y_true,
        predicted_mean,
        predicted_uncertainty,
    )

    absolute_error = np.abs(
        y_true - predicted_mean
    )

    (
        pearson_error_correlation,
        spearman_error_correlation,
    ) = _safe_correlations(
        predicted_uncertainty,
        absolute_error,
    )

    uncertainty_bin_statistics = (
        calculate_uncertainty_bins(
            y_true=y_true,
            predicted_mean=predicted_mean,
            predicted_uncertainty=(
                predicted_uncertainty
            ),
            n_bins=n_bins,
        )
    )

    if tree_predictions is None:
        interval_statistics = pd.DataFrame(
            columns=[
                "nominal_coverage",
                "lower_quantile",
                "upper_quantile",
                "observed_coverage",
                "coverage_gap",
                "absolute_coverage_gap",
                "mean_interval_width",
                "median_interval_width",
                "minimum_interval_width",
                "maximum_interval_width",
            ]
        )

    else:
        interval_statistics = (
            calculate_empirical_interval_statistics(
                y_true=y_true,
                tree_predictions=tree_predictions,
                interval_levels=interval_levels,
            )
        )

    return RegressionUncertaintyResult(
        pearson_error_correlation=(
            pearson_error_correlation
        ),
        spearman_error_correlation=(
            spearman_error_correlation
        ),
        n_samples=len(y_true),
        requested_bins=n_bins,
        effective_bins=len(
            uncertainty_bin_statistics
        ),
        uncertainty_bin_statistics=(
            uncertainty_bin_statistics
        ),
        interval_statistics=(
            interval_statistics
        ),
    )


def _validate_tree_predictions(
    tree_predictions: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Validate a matrix of ensemble predictions.

    Expected shape:

        (n_samples, n_estimators)
    """
    tree_predictions = np.asarray(
        tree_predictions,
        dtype=float,
    )

    if tree_predictions.ndim != 2:
        raise ValueError(
            "tree_predictions must be a two-dimensional array "
            "with shape (n_samples, n_estimators)."
        )

    if tree_predictions.shape[0] != n_samples:
        raise ValueError(
            "tree_predictions and y_true must contain the "
            "same number of samples."
        )

    if tree_predictions.shape[1] < 2:
        raise ValueError(
            "At least two ensemble predictions per sample "
            "are required."
        )

    if not np.isfinite(
        tree_predictions
    ).all():
        raise ValueError(
            "tree_predictions contains non-finite values."
        )

    return tree_predictions