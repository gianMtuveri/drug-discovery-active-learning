import argparse

import numpy as np

from src.active_learning.batch_selection import (
    TopKBatchSelector,
)
from src.active_learning.criteria import (
    AcquisitionContext,
)
from src.active_learning.factories import (
    make_prediction_engine,
    make_ucb_engine,
    make_uncertainty_engine,
)
from src.data.targets import load_target_regression
from src.regression.models import make_regression_model
from src.regression.selection import (
    select_greedy,
    select_ucb,
    select_uncertainty,
)
from src.regression.uncertainty import (
    predict_with_uncertainty,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default="EGFR",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n-initial",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 2.0],
    )

    return parser.parse_args()


def assert_same_selection(
    *,
    name: str,
    old_selection: np.ndarray,
    new_selection: np.ndarray,
) -> None:
    same = np.array_equal(
        old_selection,
        new_selection,
    )

    print(
        f"{name:28s} "
        f"equal={same}"
    )

    if not same:
        print(
            "Old:",
            old_selection,
        )
        print(
            "New:",
            new_selection,
        )

        raise AssertionError(
            f"{name} selections differ."
        )


def evaluate_engine(
    *,
    engine,
    context: AcquisitionContext,
    batch_size: int,
) -> np.ndarray:
    acquisition_result = engine.evaluate(
        context
    )

    selection_result = TopKBatchSelector().select(
        acquisition_result,
        batch_size=batch_size,
    )

    return selection_result.selected_indices


def main():
    args = parse_args()

    X, y = load_target_regression(
        args.target
    )

    rng = np.random.default_rng(
        args.seed
    )

    all_indices = np.arange(len(y))

    labeled_indices = rng.choice(
        all_indices,
        size=min(
            args.n_initial,
            len(all_indices),
        ),
        replace=False,
    )

    unlabeled_indices = np.setdiff1d(
        all_indices,
        labeled_indices,
        assume_unique=False,
    )

    model = make_regression_model(
        model_name="random_forest",
        random_state=args.seed,
    )

    model.fit(
        X[labeled_indices],
        y[labeled_indices],
    )

    predicted_mean, predicted_uncertainty = (
        predict_with_uncertainty(
            model,
            X[unlabeled_indices],
        )
    )

    context = AcquisitionContext(
        unlabeled_indices=unlabeled_indices,
        predicted_mean=predicted_mean,
        predicted_uncertainty=(
            predicted_uncertainty
        ),
        X_pool=X[unlabeled_indices],
    )

    old_greedy = select_greedy(
        unlabeled_indices,
        predicted_mean,
        args.batch_size,
    )

    new_greedy = evaluate_engine(
        engine=make_prediction_engine(),
        context=context,
        batch_size=args.batch_size,
    )

    assert_same_selection(
        name="Prediction-only vs greedy",
        old_selection=old_greedy,
        new_selection=new_greedy,
    )

    old_uncertainty = select_uncertainty(
        unlabeled_indices,
        predicted_uncertainty,
        args.batch_size,
    )

    new_uncertainty = evaluate_engine(
        engine=make_uncertainty_engine(),
        context=context,
        batch_size=args.batch_size,
    )

    assert_same_selection(
        name=(
            "Uncertainty-only vs uncertainty"
        ),
        old_selection=old_uncertainty,
        new_selection=new_uncertainty,
    )

    for beta in args.betas:
        old_ucb = select_ucb(
            unlabeled_indices=unlabeled_indices,
            predicted_affinity=(
                predicted_mean
            ),
            uncertainty=(
                predicted_uncertainty
            ),
            batch_size=args.batch_size,
            beta=beta,
        )

        new_ucb = evaluate_engine(
            engine=make_ucb_engine(
                beta=beta,
            ),
            context=context,
            batch_size=args.batch_size,
        )

        assert_same_selection(
            name=f"UCB beta={beta:g}",
            old_selection=old_ucb,
            new_selection=new_ucb,
        )

    print(
        "\nAll modular acquisition "
        "equivalence tests passed."
    )


if __name__ == "__main__":
    main()