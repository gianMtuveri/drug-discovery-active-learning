import numpy as np

from src.diagnostics.campaign_convergence import (
    combine_metric_convergence,
    evaluate_campaign_convergence,
)


def print_campaign_result(
    name: str,
    result,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Policy: {result.policy}"
    )

    print(
        "Converged metrics: "
        f"{result.n_converged_metrics}/"
        f"{result.n_metrics}"
    )

    print(
        "Convergence fraction: "
        f"{result.convergence_fraction:.3f}"
    )

    print(
        "Campaign converged: "
        f"{result.campaign_converged}"
    )

    print(
        "Converged metric names: "
        f"{result.converged_metrics}"
    )

    print(
        "Unconverged metric names: "
        f"{result.unconverged_metrics}"
    )

    print(
        result.metric_summary.to_string(
            index=False
        )
    )


def build_mixed_campaign():
    rounds = np.arange(0, 8)

    metric_values = {
        "rmse": np.array(
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
        ),
        "top20_mean_discovered": np.array(
            [
                7.20,
                7.80,
                8.20,
                8.55,
                8.75,
                8.90,
                9.05,
                9.20,
            ]
        ),
        "best_discovered": np.array(
            [
                8.00,
                9.00,
                9.60,
                10.20,
                10.20,
                10.20,
                10.20,
                10.20,
            ]
        ),
    }

    metric_directions = {
        "rmse": "minimize",
        "top20_mean_discovered": (
            "maximize"
        ),
        "best_discovered": "maximize",
    }

    absolute_tolerances = {
        "rmse": 0.01,
        "top20_mean_discovered": 0.05,
        "best_discovered": 0.01,
    }

    relative_tolerances = {
        "rmse": 0.01,
        "top20_mean_discovered": 0.01,
        "best_discovered": 0.01,
    }

    return evaluate_campaign_convergence(
        rounds=rounds,
        metric_values=metric_values,
        metric_directions=(
            metric_directions
        ),
        absolute_tolerances=(
            absolute_tolerances
        ),
        relative_tolerances=(
            relative_tolerances
        ),
        patience=3,
        tolerance_logic="all",
        maximum_iterations=10,
        policy="all",
    )


def test_all_policy() -> None:
    campaign_result, metric_results = (
        build_mixed_campaign()
    )

    print_campaign_result(
        "All-metrics policy",
        campaign_result,
    )

    if campaign_result.n_metrics != 3:
        raise AssertionError(
            "Expected three monitored metrics."
        )

    if (
        campaign_result
        .n_converged_metrics
        != 2
    ):
        raise AssertionError(
            "RMSE and best_discovered should "
            "be converged."
        )

    if not np.isclose(
        campaign_result
        .convergence_fraction,
        2 / 3,
    ):
        raise AssertionError(
            "Expected a convergence fraction "
            "of 2/3."
        )

    if campaign_result.campaign_converged:
        raise AssertionError(
            "The all-metrics policy should fail "
            "while discovery Top-20 is improving."
        )

    if set(
        campaign_result.converged_metrics
    ) != {
        "rmse",
        "best_discovered",
    }:
        raise AssertionError(
            "Unexpected converged metrics."
        )

    if set(
        campaign_result.unconverged_metrics
    ) != {
        "top20_mean_discovered"
    }:
        raise AssertionError(
            "Unexpected unconverged metrics."
        )



def build_metric_results():
    _, metric_results = build_mixed_campaign()
    return metric_results


def test_any_policy() -> None:
    metric_results = build_metric_results()
    result = combine_metric_convergence(
        list(
            metric_results.values()
        ),
        policy="any",
    )

    print_campaign_result(
        "Any-metric policy",
        result,
    )

    if not result.campaign_converged:
        raise AssertionError(
            "The any-metric policy should pass."
        )


def test_at_least_n_policy() -> None:
    metric_results = build_metric_results()
    result = combine_metric_convergence(
        list(
            metric_results.values()
        ),
        policy="at_least_n",
        minimum_converged=2,
    )

    print_campaign_result(
        "At-least-two policy",
        result,
    )

    if not result.campaign_converged:
        raise AssertionError(
            "Two converged metrics should satisfy "
            "the at-least-two policy."
        )

    strict_result = (
        combine_metric_convergence(
            list(
                metric_results.values()
            ),
            policy="at_least_n",
            minimum_converged=3,
        )
    )

    if strict_result.campaign_converged:
        raise AssertionError(
            "Three required metrics should fail "
            "when only two are converged."
        )


def test_mapping_validation() -> None:
    rounds = np.arange(0, 4)

    try:
        evaluate_campaign_convergence(
            rounds=rounds,
            metric_values={
                "rmse": [
                    1.5,
                    1.2,
                    1.1,
                    1.0,
                ]
            },
            metric_directions={
                "rmse": "minimize",
            },
            absolute_tolerances={
                "rmse": 0.01,
            },
            relative_tolerances={
                "wrong_name": 0.01,
            },
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Mismatched metric configuration "
            "should raise ValueError."
        )


def main() -> None:
    test_all_policy()
    test_any_policy()
    test_at_least_n_policy()
    test_mapping_validation()

    print(
        "\nAll campaign convergence tests passed."
    )


if __name__ == "__main__":
    main()