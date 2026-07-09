import numpy as np
import pandas as pd

from src.chemical_space import compute_pca_embedding
from src.simulation import run_simulation
from src.plotting import plot_embedding_campaign_comparison


def main():
    X = np.load("data/processed/bindingdb_egfr_morgan_X.npy")
    y = np.load("data/processed/bindingdb_egfr_y.npy")

    summary = pd.read_csv(
        "results/tables/bindingdb_egfr_initialization_comparison_summary.csv"
    )

    embedding, pca = compute_pca_embedding(X)
    explained = pca.explained_variance_ratio_

    histories = {}

    for initialization_strategy in ["random", "diverse"]:
        for selection_strategy in [
            "random",
            "greedy",
            "uncertainty_topk",
            "uncertainty_diverse",
            "query_by_committee"
        ]:
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

    title = (
        "BindingDB EGFR PCA chemical-space campaign "
        f"(PC1={explained[0]:.1%}, PC2={explained[1]:.1%})"
    )

    plot_embedding_campaign_comparison(
        embedding=embedding,
        y=y,
        histories=histories,
        summary=summary,
        target_round=10,
        output_path="results/figures/bindingdb_egfr_pca_campaign_comparison_round10.png",
        title=title,
    )


if __name__ == "__main__":
    main()