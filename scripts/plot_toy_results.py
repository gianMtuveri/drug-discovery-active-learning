import pandas as pd
from sklearn.model_selection import train_test_split

from src.toy_data import make_toy_dataset
from src.pool import initialize_pool
from src.plotting import plot_roc_auc, plot_2d_pool


def main():
    summary = pd.read_csv("results/toy_repeated_summary.csv")
    plot_roc_auc(summary)

    X, y = make_toy_dataset(n_samples=1000, random_state=42)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    labeled_indices, unlabeled_indices = initialize_pool(
        n_samples=len(y_pool),
        n_initial=20,
        random_state=42,
    )

    for strategy in ["random", "greedy", "uncertainty_topk", "uncertainty_diverse", "query_by_committee"]:
        plot_2d_pool(
            X_pool=X_pool,
            y_pool=y_pool,
            labeled_indices=labeled_indices,
            strategy=strategy,
            output_path=f"results/figures/toy_2d_initial_{strategy}.png",
        )


if __name__ == "__main__":
    main()