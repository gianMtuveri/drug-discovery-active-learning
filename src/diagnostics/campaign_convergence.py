from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from src.diagnostics.convergence import (
    Direction,
    evaluate_metric_convergence,
)
from src.diagnostics.results import (
    CampaignConvergenceResult,
    ConvergenceResult,
    MetricConvergenceSummary,
)


CampaignPolicy = Literal[
    "all",
    "any",
    "at_least_n",
]


def _validate_policy(
    *,
    policy: CampaignPolicy,
    n_metrics: int,
    minimum_converged: int | None,
) -> int | None:
    """
    Validate and normalize a campaign convergence policy.
    """
    valid_policies = {
        "all",
        "any",
        "at_least_n",
    }

    if policy not in valid_policies:
        raise ValueError(
            f"Unknown policy '{policy}'. "
            f"Expected one of {sorted(valid_policies)}."
        )

    if n_metrics <= 0:
        raise ValueError(
            "At least one metric is required."
        )

    if policy == "at_least_n":
        if minimum_converged is None:
            raise ValueError(
                "minimum_converged is required when "
                "policy='at_least_n'."
            )

        if not 1 <= minimum_converged <= n_metrics:
            raise ValueError(
                "minimum_converged must be between 1 and "
                "the number of monitored metrics."
            )

        return int(minimum_converged)

    if minimum_converged is not None:
        raise ValueError(
            "minimum_converged should only be provided when "
            "policy='at_least_n'."
        )

    return None


def _apply_campaign_policy(
    *,
    policy: CampaignPolicy,
    n_converged: int,
    n_metrics: int,
    minimum_converged: int | None,
) -> bool:
    """
    Combine metric-level convergence states using an explicit policy.
    """
    if policy == "all":
        return n_converged == n_metrics

    if policy == "any":
        return n_converged >= 1

    if minimum_converged is None:
        raise RuntimeError(
            "minimum_converged was not validated."
        )

    return n_converged >= minimum_converged


def summarize_metric_convergence(
    result: ConvergenceResult,
) -> MetricConvergenceSummary:
    """
    Convert a full metric result into a compact campaign summary.
    """
    return MetricConvergenceSummary(
        metric_name=result.metric_name,
        direction=result.direction,
        currently_converged=(
            result.currently_converged
        ),
        ever_converged=result.ever_converged,
        first_convergence_round=(
            result.first_convergence_round
        ),
        current_plateau_start_round=(
            result.current_plateau_start_round
        ),
        current_stable_rounds=(
            result.current_stable_rounds
        ),
        best_round=result.best_round,
        best_value=result.best_value,
        termination_status=(
            result.termination_status
        ),
    )


def combine_metric_convergence(
    metric_results: Sequence[ConvergenceResult],
    *,
    policy: CampaignPolicy = "all",
    minimum_converged: int | None = None,
) -> CampaignConvergenceResult:
    """
    Combine independently evaluated metric trajectories.

    Parameters
    ----------
    metric_results
        One convergence result per monitored metric.

    policy
        ``all`` requires every metric to be currently converged.

        ``any`` requires at least one metric to be currently converged.

        ``at_least_n`` requires a user-specified number of currently
        converged metrics.

    minimum_converged
        Required only for ``at_least_n``.

    Notes
    -----
    Metrics are not numerically averaged. Each metric first produces
    an independent Boolean convergence state. The campaign policy then
    combines those states transparently.
    """
    metric_results = list(
        metric_results
    )

    if not metric_results:
        raise ValueError(
            "At least one metric result is required."
        )

    metric_names = [
        result.metric_name
        for result in metric_results
    ]

    duplicate_names = {
        name
        for name in metric_names
        if metric_names.count(name) > 1
    }

    if duplicate_names:
        raise ValueError(
            "Metric names must be unique. "
            f"Duplicates: {sorted(duplicate_names)}"
        )

    n_metrics = len(metric_results)

    minimum_converged = _validate_policy(
        policy=policy,
        n_metrics=n_metrics,
        minimum_converged=minimum_converged,
    )

    compact_results = [
        summarize_metric_convergence(result)
        for result in metric_results
    ]

    rows = [
        {
            "metric_name": result.metric_name,
            "direction": result.direction,
            "currently_converged": (
                result.currently_converged
            ),
            "ever_converged": (
                result.ever_converged
            ),
            "first_convergence_round": (
                result.first_convergence_round
            ),
            "current_plateau_start_round": (
                result.current_plateau_start_round
            ),
            "current_stable_rounds": (
                result.current_stable_rounds
            ),
            "best_round": result.best_round,
            "best_value": result.best_value,
            "termination_status": (
                result.termination_status
            ),
        }
        for result in compact_results
    ]

    metric_summary = pd.DataFrame(
        rows
    )

    converged_metrics = tuple(
        metric_summary.loc[
            metric_summary[
                "currently_converged"
            ],
            "metric_name",
        ].tolist()
    )

    unconverged_metrics = tuple(
        metric_summary.loc[
            ~metric_summary[
                "currently_converged"
            ],
            "metric_name",
        ].tolist()
    )

    n_converged = len(
        converged_metrics
    )

    convergence_fraction = (
        n_converged / n_metrics
    )

    campaign_converged = (
        _apply_campaign_policy(
            policy=policy,
            n_converged=n_converged,
            n_metrics=n_metrics,
            minimum_converged=minimum_converged,
        )
    )

    return CampaignConvergenceResult(
        n_metrics=n_metrics,
        n_converged_metrics=n_converged,
        convergence_fraction=float(
            convergence_fraction
        ),
        policy=policy,
        minimum_converged=(
            minimum_converged
        ),
        campaign_converged=(
            campaign_converged
        ),
        converged_metrics=(
            converged_metrics
        ),
        unconverged_metrics=(
            unconverged_metrics
        ),
        metric_summary=metric_summary,
    )


def evaluate_campaign_convergence(
    *,
    rounds,
    metric_values: Mapping[str, Sequence[float]],
    metric_directions: Mapping[str, Direction],
    absolute_tolerances: Mapping[
        str,
        float | None,
    ],
    relative_tolerances: Mapping[
        str,
        float | None,
    ],
    patience: int = 3,
    tolerance_logic: str = "all",
    maximum_iterations: int | None = None,
    policy: CampaignPolicy = "all",
    minimum_converged: int | None = None,
) -> tuple[
    CampaignConvergenceResult,
    dict[str, ConvergenceResult],
]:
    """
    Evaluate several campaign metrics independently, then combine them.

    Each metric retains its own direction and tolerance configuration.
    This prevents incomparable scales such as RMSE and discovered
    affinity from being mixed into one numerical score.
    """
    metric_names = list(
        metric_values
    )

    if not metric_names:
        raise ValueError(
            "metric_values cannot be empty."
        )

    required_mappings = {
        "metric_directions": metric_directions,
        "absolute_tolerances": (
            absolute_tolerances
        ),
        "relative_tolerances": (
            relative_tolerances
        ),
    }

    for mapping_name, mapping in (
        required_mappings.items()
    ):
        missing = set(metric_names) - set(
            mapping
        )

        extra = set(mapping) - set(
            metric_names
        )

        if missing or extra:
            raise ValueError(
                f"{mapping_name} must contain exactly "
                "the same metric names as metric_values. "
                f"Missing: {sorted(missing)}; "
                f"extra: {sorted(extra)}."
            )

    metric_results: dict[
        str,
        ConvergenceResult,
    ] = {}

    for metric_name in metric_names:
        metric_results[metric_name] = (
            evaluate_metric_convergence(
                rounds=rounds,
                values=metric_values[
                    metric_name
                ],
                metric_name=metric_name,
                direction=metric_directions[
                    metric_name
                ],
                absolute_tolerance=(
                    absolute_tolerances[
                        metric_name
                    ]
                ),
                relative_tolerance=(
                    relative_tolerances[
                        metric_name
                    ]
                ),
                tolerance_logic=(
                    tolerance_logic
                ),
                patience=patience,
                maximum_iterations=(
                    maximum_iterations
                ),
            )
        )

    campaign_result = (
        combine_metric_convergence(
            list(
                metric_results.values()
            ),
            policy=policy,
            minimum_converged=(
                minimum_converged
            ),
        )
    )

    return (
        campaign_result,
        metric_results,
    )