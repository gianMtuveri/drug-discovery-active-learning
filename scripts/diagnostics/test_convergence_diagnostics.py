import numpy as np

from src.diagnostics.convergence import (
    evaluate_metric_convergence,
)


def print_result(
    name: str,
    result,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Metric: {result.metric_name}"
    )
    print(
        f"Direction: {result.direction}"
    )
    print(
        f"Best round/value: "
        f"{result.best_round} / "
        f"{result.best_value:.6f}"
    )
    print(
        f"Currently converged: "
        f"{result.currently_converged}"
    )
    print(
        f"Ever converged: "
        f"{result.ever_converged}"
    )
    print(
        f"First convergence round: "
        f"{result.first_convergence_round}"
    )
    print(
        f"Current plateau start: "
        f"{result.current_plateau_start_round}"
    )
    print(
        f"Current stable rounds: "
        f"{result.current_stable_rounds}"
    )
    print(
        f"Maximum iterations: "
        f"{result.maximum_iterations}"
    )
    print(
        f"Reached maximum iterations: "
        f"{result.reached_maximum_iterations}"
    )
    print(
        f"Termination status: "
        f"{result.termination_status}"
    )

    print(
        result.round_statistics.to_string(
            index=False
        )
    )


def test_steady_improvement() -> None:
    rounds = np.arange(0, 7)

    values = np.array(
        [
            1.50,
            1.35,
            1.22,
            1.10,
            1.00,
            0.91,
            0.83,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="rmse",
        direction="minimize",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=10,
    )

    print_result(
        "Steady improvement",
        result,
    )

    if result.currently_converged:
        raise AssertionError(
            "A steadily improving campaign "
            "must not be converged."
        )

    if result.ever_converged:
        raise AssertionError(
            "A steadily improving campaign "
            "must never register convergence."
        )

    if result.reached_maximum_iterations:
        raise AssertionError(
            "The campaign should not have reached "
            "its maximum iteration budget."
        )

    if (
        result.termination_status
        != "campaign_in_progress"
    ):
        raise AssertionError(
            "Unexpected termination status for "
            "an improving campaign."
        )


def test_clear_plateau() -> None:
    rounds = np.arange(0, 8)

    values = np.array(
        [
            1.50,
            1.25,
            1.10,
            1.005,
            1.002,
            1.001,
            1.0005,
            1.0004,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="rmse",
        direction="minimize",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=10,
    )

    print_result(
        "Clear plateau",
        result,
    )

    if not result.currently_converged:
        raise AssertionError(
            "A clear plateau should be "
            "currently converged."
        )

    if result.first_convergence_round != 6:
        raise AssertionError(
            "The expected first convergence "
            "round is 6."
        )

    if (
        result.termination_status
        != "converged_before_maximum"
    ):
        raise AssertionError(
            "The plateau should be detected "
            "before budget exhaustion."
        )


def test_temporary_plateau() -> None:
    rounds = np.arange(0, 9)

    values = np.array(
        [
            1.50,
            1.20,
            1.105,
            1.102,
            1.101,
            1.100,
            0.85,
            0.70,
            0.60,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="rmse",
        direction="minimize",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=10,
    )

    print_result(
        "Temporary plateau",
        result,
    )

    if not result.ever_converged:
        raise AssertionError(
            "The temporary plateau should "
            "register an early convergence event."
        )

    if result.currently_converged:
        raise AssertionError(
            "Strong later improvement should reset "
            "the current convergence state."
        )

    if (
        result.termination_status
        != "campaign_in_progress"
    ):
        raise AssertionError(
            "The campaign should remain in progress."
        )


def test_noisy_plateau_at_budget() -> None:
    rounds = np.arange(0, 11)

    values = np.array(
        [
            0.70,
            0.78,
            0.84,
            0.89,
            0.91,
            0.908,
            0.912,
            0.909,
            0.911,
            0.910,
            0.911,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="roc_auc",
        direction="maximize",
        absolute_tolerance=0.005,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=10,
    )

    print_result(
        "Noisy plateau at maximum budget",
        result,
    )

    if not result.currently_converged:
        raise AssertionError(
            "A noisy low-amplitude plateau "
            "should be converged."
        )

    if not result.reached_maximum_iterations:
        raise AssertionError(
            "Round 10 should satisfy the maximum "
            "iteration budget."
        )

    if (
        result.termination_status
        != "converged_at_maximum"
    ):
        raise AssertionError(
            "Expected convergence and budget "
            "exhaustion together."
        )


def test_maximum_iterations_without_convergence() -> None:
    rounds = np.arange(0, 6)

    values = np.array(
        [
            1.50,
            1.30,
            1.15,
            1.02,
            0.90,
            0.80,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="rmse",
        direction="minimize",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=5,
    )

    print_result(
        "Maximum iterations without convergence",
        result,
    )

    if result.currently_converged:
        raise AssertionError(
            "The trajectory is still improving."
        )

    if not result.reached_maximum_iterations:
        raise AssertionError(
            "The maximum iteration budget "
            "should have been reached."
        )

    if (
        result.termination_status
        != "maximum_iterations_reached"
    ):
        raise AssertionError(
            "Budget exhaustion must remain distinct "
            "from convergence."
        )


def test_major_deterioration_is_not_convergence() -> None:
    rounds = np.arange(0, 5)

    values = np.array(
        [
            1.00,
            1.05,
            1.12,
            1.20,
            1.30,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="rmse",
        direction="minimize",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=10,
    )

    print_result(
        "Major deterioration",
        result,
    )

    if result.currently_converged:
        raise AssertionError(
            "Large deterioration must not be interpreted "
            "as convergence."
        )

    if result.ever_converged:
        raise AssertionError(
            "A consistently worsening trajectory must not "
            "register a convergence event."
        )

    transition_rows = result.round_statistics.iloc[1:]

    if not transition_rows[
        "deteriorated"
    ].all():
        raise AssertionError(
            "Every transition should be marked as deterioration."
        )

    if transition_rows[
        "stable"
    ].any():
        raise AssertionError(
            "Large deteriorations should not be stable."
        )


def test_small_noise_then_major_deterioration() -> None:
    rounds = np.arange(0, 5)

    values = np.array(
        [
            1.30,
            1.305,
            1.302,
            1.304,
            1.36,
        ]
    )

    result = evaluate_metric_convergence(
        rounds=rounds,
        values=values,
        metric_name="rmse",
        direction="minimize",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        tolerance_logic="all",
        patience=3,
        maximum_iterations=10,
    )

    print_result(
        "Small noise followed by major deterioration",
        result,
    )

    if result.currently_converged:
        raise AssertionError(
            "A large final deterioration must reset "
            "the current plateau."
        )

    if result.current_stable_rounds != 0:
        raise AssertionError(
            "The large final change should reset the "
            "stable-round counter."
        )

    if result.round_statistics.iloc[-1][
        "stable"
    ]:
        raise AssertionError(
            "The final large deterioration must not "
            "be considered stable."
        )


def main() -> None:
    test_steady_improvement()
    test_clear_plateau()
    test_temporary_plateau()
    test_noisy_plateau_at_budget()
    test_maximum_iterations_without_convergence()
    test_major_deterioration_is_not_convergence()
    test_small_noise_then_major_deterioration()

    print(
        "\nAll convergence diagnostics tests passed."
    )


if __name__ == "__main__":
    main()