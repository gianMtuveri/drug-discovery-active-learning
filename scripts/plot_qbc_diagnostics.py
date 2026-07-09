import numpy as np

from src.chemical_space import compute_pca_embedding, compute_umap_embedding
from src.pool import initialize_pool
from src.committee import train_committee, compute_disagreement_scores
from src.selection import select_query_by_committee
from src.plotting import plot_qbc_diagnostics


def main():
    X = np.load("data/processed/bindingdb_egfr_morgan_X.npy")
    y = np.load("data/processed/bindingdb_egfr_y.npy")

    pca_embedding, _ = compute_pca_embedding(
        X,
        n_components=10,
        random_state=42,
    )

    umap_embedding, _ = compute_umap_embedding(
        X,
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
    )

    labeled_indices, unlabeled_indices = initialize_pool(
        X=X,
        n_initial=20,
        strategy="random",
        random_state=42,
    )

    committee = train_committee(
        X[labeled_indices],
        y[labeled_indices],
        random_state=42,
    )

    disagreement_unlabeled = compute_disagreement_scores(
        committee,
        X[unlabeled_indices],
    )

    selected_indices = select_query_by_committee(
        unlabeled_indices=unlabeled_indices,
        disagreement_scores=disagreement_unlabeled,
        batch_size=10,
    )

    # For plotting all molecules, give labeled molecules score 0.
    disagreement_all = np.zeros(len(y))
    disagreement_all[unlabeled_indices] = disagreement_unlabeled

    plot_qbc_diagnostics(
        embedding_pca=pca_embedding,
        embedding_umap=umap_embedding,
        y=y,
        disagreement_scores=disagreement_all,
        selected_indices=selected_indices,
        output_path="results/figures/bindingdb_egfr_qbc_diagnostics_round0.png",
        title="BindingDB EGFR Query by Committee diagnostics at round 0",
    )

    print("Mean disagreement:", disagreement_unlabeled.mean())
    print("Max disagreement:", disagreement_unlabeled.max())
    print("Median disagreement:", np.median(disagreement_unlabeled))


if __name__ == "__main__":
    main()