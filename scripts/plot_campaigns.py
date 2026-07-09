from sklearn.model_selection import train_test_split

from src.toy_data import make_toy_dataset
from src.simulation import run_simulation
from src.plotting import plot_campaign_view, plot_diagnostic_view


def main():
    X, y = make_toy_dataset(n_samples=1000, random_state=42)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    for strategy in ["random", "greedy", "uncertainty_topk", "uncertainty_diverse", "query_by_committee"]:
        history = run_simulation(
            X,
            y,
            strategy=strategy,
            n_initial=20,
            batch_size=10,
            n_rounds=10,
            test_size=0.2,
            random_state=42,
        )

        # Recreate the same pool used internally by run_simulation
        plot_diagnostic_view(
            X_pool=X_pool,
            y_pool=y_pool,
            labeled_indices=history.iloc[0]["initial_indices"],
            strategy=strategy,
            output_path=f"results/figures/diagnostic_round0_{strategy}.png",
        )

        plot_campaign_view(
            X_pool=X_pool,
            y_pool=y_pool,
            history=history,
            strategy=strategy,
            target_round=10,
            output_path=f"results/figures/campaign_round10_{strategy}.png",
        )


if __name__ == "__main__":
    main()