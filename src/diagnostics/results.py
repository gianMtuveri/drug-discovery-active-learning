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


@dataclass(frozen=True)
class ConvergenceResult:
    """
    Retrospective convergence analysis for one scalar metric trajectory.

    Convergence does not replace the campaign budget. The result records
    whether a plateau was detected and whether the supplied history also
    reached its configured maximum number of iterations.
    """

    metric_name: str
    direction: str

    n_observations: int
    initial_round: int | float
    final_round: int | float

    maximum_iterations: int | None
    reached_maximum_iterations: bool

    absolute_tolerance: float | None
    relative_tolerance: float | None
    tolerance_logic: str
    patience: int

    best_round: int | float
    best_value: float

    currently_converged: bool
    ever_converged: bool
    first_convergence_round: int | float | None
    current_plateau_start_round: int | float | None
    current_stable_rounds: int

    latest_absolute_improvement: float | None
    latest_relative_improvement: float | None

    termination_status: str
    round_statistics: pd.DataFrame


@dataclass(frozen=True)
class MetricConvergenceSummary:
    """
    Compact convergence summary for one monitored campaign metric.
    """

    metric_name: str
    direction: str
    currently_converged: bool
    ever_converged: bool
    first_convergence_round: int | float | None
    current_plateau_start_round: int | float | None
    current_stable_rounds: int
    best_round: int | float
    best_value: float
    termination_status: str


@dataclass(frozen=True)
class CampaignConvergenceResult:
    """
    Combined convergence summary for multiple campaign metrics.

    The convergence fraction is descriptive. The campaign-level
    decision is produced by an explicit policy rather than by averaging
    heterogeneous scientific metrics.
    """

    n_metrics: int
    n_converged_metrics: int
    convergence_fraction: float

    policy: str
    minimum_converged: int | None
    campaign_converged: bool

    converged_metrics: tuple[str, ...]
    unconverged_metrics: tuple[str, ...]

    metric_summary: pd.DataFrame