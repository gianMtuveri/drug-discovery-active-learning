from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.diagnostics.results import (
    ClassificationCalibrationResult,
)


BinningMethod = Literal[
    "uniform",
    "quantile",
    "equal_frequency",
]


def _validate_binary_inputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert binary labels and positive-class probabilities.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if probabilities.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional."
        )

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Calibration requires at least one sample."
        )

    unique_labels = np.unique(y_true)

    if not np.isin(
        unique_labels,
        [0, 1],
    ).all():
        raise ValueError(
            "y_true must contain only binary labels 0 and 1."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "probabilities contains non-finite values."
        )

    if np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError(
            "probabilities must lie between 0 and 1."
        )

    return (
        y_true.astype(int),
        probabilities,
    )


def _uniform_bin_edges(
    n_bins: int,
) -> np.ndarray:
    """
    Create equally spaced probability-bin edges.
    """
    return np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )


def _equal_frequency_bin_edges(
    probabilities: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Create approximately equal-frequency probability bins.

    Bin boundaries are calculated from prediction quantiles. When
    probabilities contain repeated values, multiple quantiles may
    collapse onto the same boundary. In that situation, fewer than
    the requested number of bins are returned rather than inventing
    artificial boundaries.

    Equal-frequency bins tend to contain similar numbers of samples,
    but may provide poor resolution when predictions have only a few
    distinct values.
    """
    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    raw_edges = np.quantile(
        probabilities,
        quantiles,
    )

    edges = np.unique(raw_edges)

    if len(edges) < 2:
        # All predictions are identical.
        value = float(probabilities[0])

        lower = max(
            0.0,
            value - 1e-12,
        )

        upper = min(
            1.0,
            value + 1e-12,
        )

        if np.isclose(lower, upper):
            lower = 0.0
            upper = 1.0

        edges = np.array(
            [lower, upper],
            dtype=float,
        )

    # Ensure the entire valid probability domain is represented.
    edges[0] = 0.0
    edges[-1] = 1.0

    actual_bins = len(edges) - 1

    if actual_bins < n_bins:
        warnings.warn(
            (
                "Equal-frequency binning produced "
                f"{actual_bins} bin(s) instead of the requested "
                f"{n_bins}. This occurs when predicted probabilities "
                "contain many duplicate values. Calibration metrics "
                "such as ECE may lose resolution in this case."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    return edges


def _make_bin_edges(
    probabilities: np.ndarray,
    n_bins: int,
    binning: BinningMethod,
) -> np.ndarray:
    """
    Construct calibration-bin boundaries.

    ``uniform`` divides the probability interval [0, 1] into equally
    wide bins. It preserves resolution across the probability scale,
    but may produce sparsely populated or empty bins.

    ``equal_frequency`` uses prediction quantiles so bins contain
    approximately equal sample counts. It is often useful with
    imbalanced prediction distributions, but repeated probabilities
    can collapse several requested bins into one.

    ``quantile`` is retained as an alias for ``equal_frequency``.
    """
    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    if binning == "uniform":
        return _uniform_bin_edges(
            n_bins
        )

    if binning in {
        "quantile",
        "equal_frequency",
    }:
        return _equal_frequency_bin_edges(
            probabilities,
            n_bins,
        )

    raise ValueError(
        "binning must be 'uniform', 'equal_frequency', "
        "or the backwards-compatible alias 'quantile'."
    )


def calculate_calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
    binning: BinningMethod = "equal_frequency",
) -> pd.DataFrame:
    """
    Calculate bin-level binary calibration statistics.

    Parameters
    ----------
    y_true
        Binary observed outcomes.
    probabilities
        Predicted probabilities for the positive class.
    n_bins
        Requested number of calibration bins.
    binning
        ``uniform`` creates equal-width intervals over [0, 1].

        ``equal_frequency`` creates bins from prediction quantiles so
        that sample counts are approximately balanced. It may return
        fewer bins when predictions contain repeated values.

        ``quantile`` is an alias for ``equal_frequency``.

    Returns
    -------
    pandas.DataFrame
        One row per effective bin, including sample count, predicted
        probability, observed positive fraction, and calibration gap.
    """
    y_true, probabilities = (
        _validate_binary_inputs(
            y_true,
            probabilities,
        )
    )

    edges = _make_bin_edges(
        probabilities,
        n_bins,
        binning,
    )

    # np.digitize returns bin positions from 0 to n_bins - 1.
    # Internal edges are sufficient because probabilities are
    # already constrained to [0, 1].
    bin_indices = np.digitize(
        probabilities,
        edges[1:-1],
        right=True,
    )

    rows: list[dict[str, float | int]] = []
    n_samples = len(y_true)

    for bin_index in range(
        len(edges) - 1
    ):
        mask = bin_indices == bin_index
        count = int(mask.sum())

        lower_bound = float(
            edges[bin_index]
        )
        upper_bound = float(
            edges[bin_index + 1]
        )

        if count == 0:
            mean_probability = np.nan
            observed_fraction = np.nan
            signed_gap = np.nan
            absolute_gap = np.nan
        else:
            mean_probability = float(
                np.mean(
                    probabilities[mask]
                )
            )

            observed_fraction = float(
                np.mean(
                    y_true[mask]
                )
            )

            signed_gap = (
                observed_fraction
                - mean_probability
            )

            absolute_gap = abs(
                signed_gap
            )

        rows.append(
            {
                "bin_index": bin_index,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": count,
                "fraction": (
                    count / n_samples
                ),
                "mean_probability": (
                    mean_probability
                ),
                "observed_positive_fraction": (
                    observed_fraction
                ),
                "signed_calibration_gap": (
                    signed_gap
                ),
                "absolute_calibration_gap": (
                    absolute_gap
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_classification_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
    binning: BinningMethod = "equal_frequency",
    probability_clip: float = 1e-15,
) -> ClassificationCalibrationResult:
    """
    Evaluate binary probability calibration.

    Brier score and log loss evaluate probability quality at the
    individual-sample level.

    Expected Calibration Error is the sample-weighted mean absolute
    difference between observed event frequency and mean predicted
    probability in each bin.

    Maximum Calibration Error is the largest absolute bin gap.

    ECE and MCE depend on the chosen binning scheme. In particular,
    equal-frequency binning can lose resolution when predictions
    contain only a small number of distinct probability values.
    """

    canonical_binning = (
        "equal_frequency"
        if binning == "quantile"
        else binning
    )


    y_true, probabilities = (
        _validate_binary_inputs(
            y_true,
            probabilities,
        )
    )

    if not 0.0 < probability_clip < 0.5:
        raise ValueError(
            "probability_clip must be between 0 and 0.5."
        )

    bin_statistics = (
        calculate_calibration_bins(
            y_true=y_true,
            probabilities=probabilities,
            n_bins=n_bins,
            binning=binning,
        )
    )

    nonempty_bins = bin_statistics[
        bin_statistics["count"] > 0
    ]

    expected_calibration_error = float(
        np.sum(
            nonempty_bins["fraction"]
            * nonempty_bins[
                "absolute_calibration_gap"
            ]
        )
    )

    maximum_calibration_error = float(
        nonempty_bins[
            "absolute_calibration_gap"
        ].max()
    )

    clipped_probabilities = np.clip(
        probabilities,
        probability_clip,
        1.0 - probability_clip,
    )

    return ClassificationCalibrationResult(
        brier_score=float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        log_loss=float(
            log_loss(
                y_true,
                clipped_probabilities,
                labels=[0, 1],
            )
        ),
        expected_calibration_error=(
            expected_calibration_error
        ),
        maximum_calibration_error=(
            maximum_calibration_error
        ),
        n_bins=len(bin_statistics),
        binning=canonical_binning,
        n_samples=len(y_true),
        bin_statistics=bin_statistics,
    )