import warnings

import numpy as np

from src.diagnostics.classification_calibration import (
    evaluate_classification_calibration,
)
from src.diagnostics.regression_calibration import (
    evaluate_regression_uncertainty,
)


def print_classification_result(
    name: str,
    result,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Brier score: "
        f"{result.brier_score:.4f}"
    )
    print(
        f"Log loss: "
        f"{result.log_loss:.4f}"
    )
    print(
        f"ECE: "
        f"{result.expected_calibration_error:.4f}"
    )
    print(
        f"MCE: "
        f"{result.maximum_calibration_error:.4f}"
    )

    print(result.bin_statistics)


def test_classification_calibration() -> None:
    rng = np.random.default_rng(42)

    probabilities = rng.uniform(
        0.0,
        1.0,
        size=20_000,
    )

    calibrated_labels = rng.binomial(
        n=1,
        p=probabilities,
    )

    overconfident_probabilities = np.where(
        probabilities >= 0.5,
        0.99,
        0.01,
    )

    constant_probabilities = np.full(
        len(calibrated_labels),
        calibrated_labels.mean(),
    )

    for binning in [
        "uniform",
        "equal_frequency",
    ]:
        print(
            f"\n\nCLASSIFICATION BINNING: "
            f"{binning.upper()}"
        )

        calibrated_result = (
            evaluate_classification_calibration(
                calibrated_labels,
                probabilities,
                n_bins=10,
                binning=binning,
            )
        )

        if binning == "equal_frequency":
            with warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter("always")

                overconfident_result = (
                    evaluate_classification_calibration(
                        calibrated_labels,
                        overconfident_probabilities,
                        n_bins=10,
                        binning=binning,
                    )
                )

            warning_found = any(
                "instead of the requested"
                in str(warning.message)
                for warning in caught
            )

            if not warning_found:
                raise AssertionError(
                    "Collapsed equal-frequency bins "
                    "should emit a warning."
                )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    RuntimeWarning,
                )

                constant_result = (
                    evaluate_classification_calibration(
                        calibrated_labels,
                        constant_probabilities,
                        n_bins=10,
                        binning=binning,
                    )
                )

        else:
            overconfident_result = (
                evaluate_classification_calibration(
                    calibrated_labels,
                    overconfident_probabilities,
                    n_bins=10,
                    binning=binning,
                )
            )

            constant_result = (
                evaluate_classification_calibration(
                    calibrated_labels,
                    constant_probabilities,
                    n_bins=10,
                    binning=binning,
                )
            )

        print_classification_result(
            "Calibrated probabilities",
            calibrated_result,
        )
        print_classification_result(
            "Overconfident probabilities",
            overconfident_result,
        )
        print_classification_result(
            "Constant base-rate probabilities",
            constant_result,
        )

        if binning == "uniform":
            if not (
                calibrated_result
                .expected_calibration_error
                <
                overconfident_result
                .expected_calibration_error
            ):
                raise AssertionError(
                    "With uniform binning, calibrated "
                    "probabilities should have lower ECE "
                    "than overconfident probabilities."
                )

        if not (
            calibrated_result.brier_score
            <
            overconfident_result.brier_score
        ):
            raise AssertionError(
                "Calibrated probabilities should have "
                "a lower Brier score."
            )

        if not (
            calibrated_result.log_loss
            <
            overconfident_result.log_loss
        ):
            raise AssertionError(
                "Calibrated probabilities should have "
                "lower log loss."
            )

        if (
            binning == "equal_frequency"
            and len(
                overconfident_result.bin_statistics
            ) > 2
        ):
            raise AssertionError(
                "Equal-frequency binning should collapse "
                "to very few bins for two-valued predictions."
            )

    print(
        "\nAll classification calibration tests passed."
    )


def print_regression_result(
    name: str,
    result,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(
        "Pearson uncertainty-error correlation: "
        f"{result.pearson_error_correlation:.4f}"
    )
    print(
        "Spearman uncertainty-error correlation: "
        f"{result.spearman_error_correlation:.4f}"
    )
    print(
        "Effective uncertainty bins: "
        f"{result.effective_bins}/"
        f"{result.requested_bins}"
    )

    print(
        result.uncertainty_bin_statistics
    )

    if not result.interval_statistics.empty:
        print(
            "\nEmpirical interval coverage"
        )

        print(
            result.interval_statistics.to_string(index=False)
        )


def test_regression_uncertainty() -> None:
    rng = np.random.default_rng(42)
    n_samples = 4_000

    y_true = rng.normal(
        loc=7.0,
        scale=1.0,
        size=n_samples,
    )

    informative_uncertainty = np.linspace(
        0.05,
        1.5,
        n_samples,
    )

    error_sign = rng.choice(
        [-1.0, 1.0],
        size=n_samples,
    )

    informative_absolute_error = (
        informative_uncertainty
        + rng.normal(
            loc=0.0,
            scale=0.03,
            size=n_samples,
        )
    )

    informative_absolute_error = np.clip(
        informative_absolute_error,
        0.0,
        None,
    )

    informative_prediction = (
        y_true
        + error_sign
        * informative_absolute_error
    )

    informative_result = (
        evaluate_regression_uncertainty(
            y_true=y_true,
            predicted_mean=(
                informative_prediction
            ),
            predicted_uncertainty=(
                informative_uncertainty
            ),
            n_bins=4,
        )
    )

    # Separate synthetic experiment for empirical interval coverage.
    #
    # The observed values and ensemble predictions are independent
    # draws from the same predictive distribution. This makes
    # nominal and observed interval coverage directly comparable.
    n_interval_samples = 20_000
    n_trees = 500

    latent_mean = rng.normal(
        loc=7.0,
        scale=1.0,
        size=n_interval_samples,
    )

    interval_uncertainty = rng.uniform(
        low=0.1,
        high=1.0,
        size=n_interval_samples,
    )

    interval_y_true = (
        latent_mean
        + rng.normal(
            loc=0.0,
            scale=interval_uncertainty,
            size=n_interval_samples,
        )
    )

    interval_tree_predictions = (
        latent_mean[:, None]
        + rng.normal(
            loc=0.0,
            scale=interval_uncertainty[:, None],
            size=(
                n_interval_samples,
                n_trees,
            ),
        )
    )

    interval_predicted_mean = np.mean(
        interval_tree_predictions,
        axis=1,
    )

    interval_predicted_uncertainty = np.std(
        interval_tree_predictions,
        axis=1,
    )

    interval_result = (
        evaluate_regression_uncertainty(
            y_true=interval_y_true,
            predicted_mean=(
                interval_predicted_mean
            ),
            predicted_uncertainty=(
                interval_predicted_uncertainty
            ),
            tree_predictions=(
                interval_tree_predictions
            ),
            n_bins=4,
        )
    )

    random_uncertainty = rng.permutation(
        informative_uncertainty
    )

    random_result = (
        evaluate_regression_uncertainty(
            y_true=y_true,
            predicted_mean=(
                informative_prediction
            ),
            predicted_uncertainty=(
                random_uncertainty
            ),
            n_bins=4,
        )
    )

    constant_uncertainty = np.full(
        n_samples,
        0.5,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter("always")

        constant_result = (
            evaluate_regression_uncertainty(
                y_true=y_true,
                predicted_mean=(
                    informative_prediction
                ),
                predicted_uncertainty=(
                    constant_uncertainty
                ),
                n_bins=4,
            )
        )

    if not any(
        "instead of the requested"
        in str(warning.message)
        for warning in caught
    ):
        raise AssertionError(
            "Constant uncertainty should emit a "
            "collapsed-bin warning."
        )

    print("\n\nREGRESSION UNCERTAINTY")
    print_regression_result(
        "Informative uncertainty",
        informative_result,
    )
    print_regression_result(
        "Calibrated empirical intervals",
        interval_result,
    )
    print_regression_result(
        "Randomly permuted uncertainty",
        random_result,
    )
    print_regression_result(
        "Constant uncertainty",
        constant_result,
    )

    if (
        informative_result
        .pearson_error_correlation
        < 0.95
    ):
        raise AssertionError(
            "Informative uncertainty should have "
            "a high Pearson correlation with error."
        )

    if (
        informative_result
        .spearman_error_correlation
        < 0.95
    ):
        raise AssertionError(
            "Informative uncertainty should have "
            "a high Spearman correlation with error."
        )

    if abs(
        random_result
        .spearman_error_correlation
    ) > 0.10:
        raise AssertionError(
            "Random uncertainty should have near-zero "
            "rank correlation with error."
        )

    informative_bins = (
        informative_result
        .uncertainty_bin_statistics
    )

    if not (
        informative_bins.iloc[-1]["mae"]
        >
        informative_bins.iloc[0]["mae"]
    ):
        raise AssertionError(
            "The highest-uncertainty bin should have "
            "larger MAE than the lowest-uncertainty bin."
        )

    if not np.isnan(
        constant_result
        .pearson_error_correlation
    ):
        raise AssertionError(
            "Pearson correlation should be NaN "
            "for constant uncertainty."
        )

    if not np.isnan(
        constant_result
        .spearman_error_correlation
    ):
        raise AssertionError(
            "Spearman correlation should be NaN "
            "for constant uncertainty."
        )

    if constant_result.effective_bins != 1:
        raise AssertionError(
            "Constant uncertainty should produce "
            "one effective bin."
        )
    
    interval_statistics = (
        interval_result
        .interval_statistics
    )

    if interval_statistics.empty:
        raise AssertionError(
            "Tree predictions should produce "
            "interval statistics."
        )

    if not (
        interval_statistics[
            "observed_coverage"
        ].is_monotonic_increasing
    ):
        raise AssertionError(
            "Observed coverage should increase "
            "with nominal interval level."
        )

    if not (
        interval_statistics[
            "mean_interval_width"
        ].is_monotonic_increasing
    ):
        raise AssertionError(
            "Interval width should increase "
            "with nominal interval level."
        )

    if (
        interval_statistics[
            "absolute_coverage_gap"
        ].max()
        > 0.05
    ):
        raise AssertionError(
            "Synthetic calibrated intervals should "
            "remain close to nominal coverage."
        )

    print(
        "\nAll regression uncertainty tests passed."
    )


def main() -> None:
    test_classification_calibration()
    test_regression_uncertainty()

    print(
        "\nAll calibration diagnostics tests passed."
    )


if __name__ == "__main__":
    main()
