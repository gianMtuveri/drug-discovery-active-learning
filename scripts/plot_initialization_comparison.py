import pandas as pd
from sklearn.model_selection import train_test_split

from src.toy_data import make_toy_dataset
from src.simulation import run_simulation
from src.plotting import plot_campaign_initialization_comparison


def main():
    X, y = make_toy_dataset(n_samples=1000, random_state=42)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    summary = pd.read_csv(
        "results/tables/toy_initialization_comparison_summary.csv"
    )

    histories = {}

    for initialization_strategy in ["random", "diverse"]:
        for selection_strategy in ["random", "greedy", "uncertainty_topk", "uncertainty_diverse", "query_by_committee"]:
            history = run_simulation(
                X,
                y,
                strategy=selection_strategy,
                initialization_strategy=initialization_strategy,
                n_initial=20,
                batch_size=10,
                n_rounds=10,
                test_size=0.2,
                random_state=42,
            )

            histories[(initialization_strategy, selection_strategy)] = history

    plot_campaign_initialization_comparison(
        X_pool=X_pool,
        y_pool=y_pool,
        histories=histories,
        summary=summary,
        target_round=10,
        output_path="results/figures/campaign_initialization_comparison_round10.png",
    )


if __name__ == "__main__":
    main()