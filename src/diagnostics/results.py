from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ClassificationCalibrationResult:
    """
    Summary and bin-level results for binary classification calibration.
    """

    brier_score: float
    log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float
    n_bins: int
    binning: str
    n_samples: int
    bin_statistics: pd.DataFrame


@dataclass(frozen=True)
class RegressionUncertaintyResult:
    """
    Relationship between regression uncertainty and prediction error,
    including empirical ensemble interval coverage.
    """

    pearson_error_correlation: float
    spearman_error_correlation: float
    n_samples: int
    requested_bins: int
    effective_bins: int
    uncertainty_bin_statistics: pd.DataFrame
    interval_statistics: pd.DataFrame