import numpy as np

from src.chemical_space import compute_pca_embedding
from src.simulation import run_simulation
from src.plotting import plot_activity_embedding, plot_campaign_on_embedding


def main():
    X = np.load("data/processed/bindingdb_egfr_morgan_X.npy")
    y = np.load("data/processed/bindingdb_egfr_y.npy")

    embedding, pca = compute_pca_embedding(X)

    explained = pca.explained_variance_ratio_

    plot_activity_embedding(
        embedding=embedding,
        y=y,
        output_path="results/figures/bindingdb_egfr_pca_activity.png",
        title=(
            "BindingDB EGFR chemical space "
            f"(PC1={explained[0]:.1%}, PC2={explained[1]:.1%})"
        ),
    )

    for strategy in ["random", "greedy", "uncertainty_topk", "uncertainty_diverse", "query_by_committee"]:
        history = run_simulation(
            X,
            y,
            strategy=strategy,
            initialization_strategy="random",
            n_initial=20,
            batch_size=10,
            n_rounds=10,
            test_size=0.2,
            random_state=42,
        )

        plot_campaign_on_embedding(
            embedding=embedding,
            y=y,
            history=history,
            target_round=10,
            output_path=f"results/figures/bindingdb_egfr_pca_campaign_{strategy}.png",
            title=f"BindingDB EGFR PCA campaign: {strategy}",
        )


if __name__ == "__main__":
    main()