import argparse
import pandas as pd

from src.targets import (
    load_target_classification,
    load_target_descriptors,
)
from src.chemical_space import compute_pca_embedding, compute_umap_embedding
from src.simulation import run_simulation
from src.plotting import plot_classification_informative_panel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    target = args.target

    X, y = load_target_classification(target)
    descriptors = load_target_descriptors(target)

    if args.summary_path is None:
        summary_path = f"results/tables/{target.lower()}_classification_summary.csv"
    else:
        summary_path = args.summary_path

    summary = pd.read_csv(summary_path)

    pca_embedding, pca = compute_pca_embedding(
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

    histories = {}

    strategies = [
        "random",
        "greedy",
        "uncertainty_topk",
        "uncertainty_diverse",
        "query_by_committee",
    ]

    for initialization_strategy in ["random", "diverse"]:
        for selection_strategy in strategies:
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

    output_path = (
        f"results/figures/{target.lower()}_classification_panel.png"
    )

    plot_classification_informative_panel(
        embedding_pca=pca_embedding,
        pca=pca,
        embedding_umap=umap_embedding,
        y=y,
        histories=histories,
        summary=summary,
        target_round=10,
        output_path=output_path,
        target_name=target,
        descriptors=descriptors,
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()