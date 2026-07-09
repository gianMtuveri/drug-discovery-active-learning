import numpy as np

from src.chemical_space import compute_pca_embedding, compute_umap_embedding
from src.plotting import plot_chemical_space_explanatory_panel


def main():
    X = np.load("data/processed/bindingdb_egfr_morgan_X.npy")
    y = np.load("data/processed/bindingdb_egfr_y.npy")

    pca_embedding, pca = compute_pca_embedding(
        X,
        n_components=10,
        random_state=42,
    )

    umap_embedding, umap_model = compute_umap_embedding(
        X,
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
    )

    plot_chemical_space_explanatory_panel(
        pca_embedding=pca_embedding,
        pca=pca,
        umap_embedding=umap_embedding,
        y=y,
        output_path="results/figures/bindingdb_egfr_projection_panel.png",
        title="BindingDB EGFR chemical-space projection panel",
    )


if __name__ == "__main__":
    main()