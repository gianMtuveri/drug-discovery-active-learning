from typing import Literal

import numpy as np
import pandas as pd

from src.diagnostics.results import ConvergenceResult


Direction = Literal[
    "maximize",
    "minimize",
]

ToleranceLogic = Literal[
    "all",
    "any",
]


def _validate_convergence_inputs(
    rounds: np.ndarray,
    values: np.ndarray,
    *,
    direction: Direction,
    absolute_tolerance: float | None,
    relative_tolerance: float | None,
    tolerance_logic: ToleranceLogic,
    patience: int,
    maximum_iterations: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate one scalar campaign trajectory.
    """
    rounds = np.asarray(rounds)
    values = np.asarray(
        values,
        dtype=float,
    )

    if rounds.ndim != 1:
        raise ValueError(
            "rounds must be one-dimensional."
        )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if len(rounds) != len(values):
        raise ValueError(
            "rounds and values must have the same length."
        )

    if len(rounds) == 0:
        raise ValueError(
            "At least one campaign observation is required."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "values contains non-finite entries."
        )

    if len(np.unique(rounds)) != len(rounds):
        raise ValueError(
            "rounds must contain unique values."
        )

    if len(rounds) > 1:
        round_differences = np.diff(rounds)

        if np.any(round_differences <= 0):
            raise ValueError(
                "rounds must be strictly increasing."
            )

    if direction not in {
        "maximize",
        "minimize",
    }:
        raise ValueError(
            "direction must be 'maximize' or 'minimize'."
        )

    if tolerance_logic not in {
        "all",
        "any",
    }:
        raise ValueError(
            "tolerance_logic must be 'all' or 'any'."
        )

    if absolute_tolerance is None and relative_tolerance is None:
        raise ValueError(
            "At least one convergence tolerance must be provided."
        )

    if (
        absolute_tolerance is not None
        and absolute_tolerance < 0.0
    ):
        raise ValueError(
            "absolute_tolerance cannot be negative."
        )

    if (
        relative_tolerance is not None
        and relative_tolerance < 0.0
    ):
        raise ValueError(
            "relative_tolerance cannot be negative."
        )

    if patience <= 0:
        raise ValueError(
            "patience must be positive."
        )

    if (
        maximum_iterations is not None
        and maximum_iterations <= 0
    ):
        raise ValueError(
            "maximum_iterations must be positive."
        )

    return rounds, values


def _direction_aware_improvement(
    current_value: float,
    previous_value: float,
    direction: Direction,
) -> float:
    """
    Return an improvement where positive always means better.
    """
    if direction == "maximize":
        return current_value - previous_value

    return previous_value - current_value


def _relative_improvement(
    absolute_improvement: float,
    previous_value: float,
) -> float:
    """
    Scale improvement by the magnitude of the previous value.

    A small epsilon prevents division by zero.
    """
    denominator = max(
        abs(previous_value),
        np.finfo(float).eps,
    )

    return absolute_improvement / denominator


def _evaluate_tolerance(
    *,
    improvement: float,
    tolerance: float | None,
) -> bool | None:
    """
    Determine whether a metric change is small enough to represent
    stable behaviour.

    Both small improvements and small deteriorations can belong to a
    noisy plateau. Large changes in either direction are not stable.
    """
    if tolerance is None:
        return None

    return abs(improvement) <= tolerance


def _combine_stability_checks(
    *,
    stable_absolute: bool | None,
    stable_relative: bool | None,
    tolerance_logic: ToleranceLogic,
) -> bool:
    """
    Combine enabled absolute and relative convergence checks.
    """
    enabled_checks = [
        check
        for check in [
            stable_absolute,
            stable_relative,
        ]
        if check is not None
    ]

    if not enabled_checks:
        raise RuntimeError(
            "No convergence checks were enabled."
        )

    if tolerance_logic == "all":
        return all(enabled_checks)

    return any(enabled_checks)


def _termination_status(
    *,
    currently_converged: bool,
    ever_converged: bool,
    reached_maximum_iterations: bool,
    maximum_iterations: int | None,
) -> str:
    """
    Summarize the current campaign state without controlling execution.
    """
    if currently_converged and reached_maximum_iterations:
        return "converged_at_maximum"

    if currently_converged:
        return "converged_before_maximum"

    if reached_maximum_iterations:
        return "maximum_iterations_reached"

    if maximum_iterations is None:
        if ever_converged:
            return "history_complete_without_current_convergence"

        return "history_complete_without_convergence"

    return "campaign_in_progress"


def evaluate_metric_convergence(
    rounds: np.ndarray,
    values: np.ndarray,
    *,
    metric_name: str,
    direction: Direction,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    tolerance_logic: ToleranceLogic = "all",
    patience: int = 3,
    maximum_iterations: int | None = None,
) -> ConvergenceResult:
    """
    Evaluate retrospective convergence for one campaign metric.

    Parameters
    ----------
    rounds
        Strictly increasing campaign round identifiers.

    values
        Metric value at every recorded round.

    metric_name
        Human-readable metric name such as ``rmse`` or ``roc_auc``.

    direction
        ``maximize`` when larger values are preferable and ``minimize``
        when smaller values are preferable.

    absolute_tolerance
        Maximum absolute improvement considered negligible.

    relative_tolerance
        Maximum relative improvement considered negligible.

    tolerance_logic
        ``all`` requires all enabled tolerance checks to be satisfied.
        ``any`` requires at least one enabled check.

    patience
        Number of consecutive stable transitions required to declare
        convergence.

    maximum_iterations
        Configured campaign limit. It remains independent of convergence
        and is recorded so budget exhaustion is not confused with plateau
        detection.

    Notes
    -----
    This function does not terminate a simulation. It only analyzes an
    existing trajectory.
    """
    if not metric_name:
        raise ValueError(
            "metric_name cannot be empty."
        )

    rounds, values = _validate_convergence_inputs(
        rounds=rounds,
        values=values,
        direction=direction,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        tolerance_logic=tolerance_logic,
        patience=patience,
        maximum_iterations=maximum_iterations,
    )

    if direction == "maximize":
        best_position = int(
            np.argmax(values)
        )
    else:
        best_position = int(
            np.argmin(values)
        )

    rows: list[dict[str, object]] = []

    consecutive_stable_rounds = 0
    first_convergence_round = None
    current_plateau_start_round = None

    best_so_far = float(values[0])

    rows.append(
        {
            "round": rounds[0],
            "value": float(values[0]),
            "previous_value": np.nan,
            "absolute_improvement": np.nan,
            "relative_improvement": np.nan,
            "improved": False,
            "deteriorated": False,
            "stable_absolute": pd.NA,
            "stable_relative": pd.NA,
            "stable": False,
            "consecutive_stable_rounds": 0,
            "convergence_reached": False,
            "best_so_far": best_so_far,
        }
    )

    for position in range(
        1,
        len(values),
    ):
        current_value = float(
            values[position]
        )

        previous_value = float(
            values[position - 1]
        )

        absolute_improvement = (
            _direction_aware_improvement(
                current_value=current_value,
                previous_value=previous_value,
                direction=direction,
            )
        )

        relative_improvement = (
            _relative_improvement(
                absolute_improvement=absolute_improvement,
                previous_value=previous_value,
            )
        )

        stable_absolute = (
            _evaluate_tolerance(
                improvement=absolute_improvement,
                tolerance=absolute_tolerance,
            )
        )

        stable_relative = (
            _evaluate_tolerance(
                improvement=relative_improvement,
                tolerance=relative_tolerance,
            )
        )

        stable = _combine_stability_checks(
            stable_absolute=stable_absolute,
            stable_relative=stable_relative,
            tolerance_logic=tolerance_logic,
        )

        if stable:
            consecutive_stable_rounds += 1

            if consecutive_stable_rounds == 1:
                current_plateau_start_round = rounds[
                    position - 1
                ]

        else:
            consecutive_stable_rounds = 0
            current_plateau_start_round = None

        convergence_reached = (
            consecutive_stable_rounds
            >= patience
        )

        if (
            convergence_reached
            and first_convergence_round is None
        ):
            first_convergence_round = rounds[
                position
            ]

        if direction == "maximize":
            best_so_far = max(
                best_so_far,
                current_value,
            )
        else:
            best_so_far = min(
                best_so_far,
                current_value,
            )

        rows.append(
            {
                "round": rounds[position],
                "value": current_value,
                "previous_value": previous_value,
                "absolute_improvement": (
                    absolute_improvement
                ),
                "relative_improvement": (
                    relative_improvement
                ),
                "improved": (
                    absolute_improvement > 0.0
                ),
                "deteriorated": (
                    absolute_improvement < 0.0
                ),
                "stable_absolute": stable_absolute,
                "stable_relative": stable_relative,
                "stable": stable,
                "consecutive_stable_rounds": (
                    consecutive_stable_rounds
                ),
                "convergence_reached": (
                    convergence_reached
                ),
                "best_so_far": best_so_far,
            }
        )

    round_statistics = pd.DataFrame(
        rows
    )

    currently_converged = bool(
        len(values) > 1
        and consecutive_stable_rounds
        >= patience
    )

    ever_converged = (
        first_convergence_round is not None
    )

    if len(values) == 1:
        latest_absolute_improvement = None
        latest_relative_improvement = None
    else:
        latest_absolute_improvement = float(
            round_statistics.iloc[-1][
                "absolute_improvement"
            ]
        )

        latest_relative_improvement = float(
            round_statistics.iloc[-1][
                "relative_improvement"
            ]
        )

    reached_maximum_iterations = bool(
        maximum_iterations is not None
        and rounds[-1] >= maximum_iterations
    )

    return ConvergenceResult(
        metric_name=metric_name,
        direction=direction,
        n_observations=len(values),
        initial_round=rounds[0],
        final_round=rounds[-1],
        maximum_iterations=maximum_iterations,
        reached_maximum_iterations=(
            reached_maximum_iterations
        ),
        absolute_tolerance=(
            absolute_tolerance
        ),
        relative_tolerance=(
            relative_tolerance
        ),
        tolerance_logic=tolerance_logic,
        patience=patience,
        best_round=rounds[best_position],
        best_value=float(
            values[best_position]
        ),
        currently_converged=(
            currently_converged
        ),
        ever_converged=ever_converged,
        first_convergence_round=(
            first_convergence_round
        ),
        current_plateau_start_round=(
            current_plateau_start_round
            if currently_converged
            else None
        ),
        current_stable_rounds=(
            consecutive_stable_rounds
        ),
        latest_absolute_improvement=(
            latest_absolute_improvement
        ),
        latest_relative_improvement=(
            latest_relative_improvement
        ),
        termination_status=_termination_status(
            currently_converged=(
                currently_converged
            ),
            ever_converged=ever_converged,
            reached_maximum_iterations=(
                reached_maximum_iterations
            ),
            maximum_iterations=(
                maximum_iterations
            ),
        ),
        round_statistics=round_statistics,
    )