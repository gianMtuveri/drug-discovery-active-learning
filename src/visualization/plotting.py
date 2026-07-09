import matplotlib.pyplot as plt
import numpy as np


def plot_roc_auc(summary, output_path="results/toy_roc_auc.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy in summary["strategy"].unique():
        df = summary[summary["strategy"] == strategy]

        ax.plot(
            df["round"],
            df["roc_auc_mean"],
            marker="o",
            label=strategy,
        )

        ax.fill_between(
            df["round"],
            df["roc_auc_mean"] - df["roc_auc_std"],
            df["roc_auc_mean"] + df["roc_auc_std"],
            alpha=0.2,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("ROC AUC")
    ax.set_title("ROC AUC over active-learning rounds")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_2d_pool(X_pool, y_pool, labeled_indices, strategy, output_path):
    unlabeled_mask = [i for i in range(len(y_pool)) if i not in set(labeled_indices)]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        X_pool[unlabeled_mask, 0],
        X_pool[unlabeled_mask, 1],
        c="lightgray",
        s=20,
        label="Unlabeled",
        alpha=0.6,
    )

    labeled_active = labeled_indices[y_pool[labeled_indices] == 1]
    labeled_inactive = labeled_indices[y_pool[labeled_indices] == 0]

    ax.scatter(
        X_pool[labeled_active, 0],
        X_pool[labeled_active, 1],
        c="green",
        s=60,
        edgecolor="black",
        label="Labeled active",
    )

    ax.scatter(
        X_pool[labeled_inactive, 0],
        X_pool[labeled_inactive, 1],
        c="blue",
        s=60,
        edgecolor="black",
        label="Labeled inactive",
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{strategy} strategy: labeled vs unlabeled molecules")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def reconstruct_campaign_state(history, target_round):
    """
    Reconstruct initial and newly selected molecules up to target_round.

    Parameters
    ----------
    history : pd.DataFrame
        Output of run_simulation().
    target_round : int
        Round to visualize.

    Returns
    -------
    initial_indices : np.ndarray
        Molecules labeled at round 0.
    newly_selected_indices : np.ndarray
        Molecules selected during active-learning rounds.
    """

    initial_indices = np.asarray(history.iloc[0]["initial_indices"])

    selected = []

    for _, row in history.iterrows():
        if row["round"] >= target_round:
            break

        selected_indices = row["selected_indices"]

        if len(selected_indices) > 0:
            selected.extend(selected_indices)

    newly_selected_indices = np.asarray(selected, dtype=int)

    return initial_indices, newly_selected_indices


def plot_campaign_view(
    X_pool,
    y_pool,
    history,
    strategy,
    target_round,
    output_path,
):
    """
    Plot what happened during the active-learning campaign.

    Colors
    ------
    Gray  : never tested
    Blue  : initial labeled molecules
    Green : newly labeled active molecules
    Red   : newly labeled inactive molecules
    """

    initial_indices, newly_selected_indices = reconstruct_campaign_state(
        history=history,
        target_round=target_round,
    )

    all_indices = np.arange(len(y_pool))
    tested_indices = np.concatenate(
        [initial_indices, newly_selected_indices]
    )

    untested_indices = np.setdiff1d(all_indices, tested_indices)

    newly_active = newly_selected_indices[y_pool[newly_selected_indices] == 1]
    newly_inactive = newly_selected_indices[y_pool[newly_selected_indices] == 0]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        X_pool[untested_indices, 0],
        X_pool[untested_indices, 1],
        c="lightgray",
        s=20,
        alpha=0.5,
        label="Untested",
    )

    ax.scatter(
        X_pool[initial_indices, 0],
        X_pool[initial_indices, 1],
        c="blue",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        label="Initial labeled",
    )

    ax.scatter(
        X_pool[newly_active, 0],
        X_pool[newly_active, 1],
        c="green",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        label="New active",
    )

    ax.scatter(
        X_pool[newly_inactive, 0],
        X_pool[newly_inactive, 1],
        c="red",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        label="New inactive",
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{strategy}: campaign state at round {target_round}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_diagnostic_view(
    X_pool,
    y_pool,
    labeled_indices,
    strategy,
    output_path,
):
    """
    Diagnostic plot showing ground-truth labels.

    This uses labels that would be unknown in a real campaign.
    """

    labeled_indices = np.asarray(labeled_indices)

    all_indices = np.arange(len(y_pool))
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)

    unlabeled_active = unlabeled_indices[y_pool[unlabeled_indices] == 1]
    unlabeled_inactive = unlabeled_indices[y_pool[unlabeled_indices] == 0]

    labeled_active = labeled_indices[y_pool[labeled_indices] == 1]
    labeled_inactive = labeled_indices[y_pool[labeled_indices] == 0]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        X_pool[unlabeled_active, 0],
        X_pool[unlabeled_active, 1],
        c="lightgreen",
        s=20,
        alpha=0.45,
        label="Active, unlabeled",
    )

    ax.scatter(
        X_pool[unlabeled_inactive, 0],
        X_pool[unlabeled_inactive, 1],
        c="lightcoral",
        s=20,
        alpha=0.45,
        label="Inactive, unlabeled",
    )

    ax.scatter(
        X_pool[labeled_active, 0],
        X_pool[labeled_active, 1],
        c="green",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        label="Active, labeled",
    )

    ax.scatter(
        X_pool[labeled_inactive, 0],
        X_pool[labeled_inactive, 1],
        c="red",
        s=70,
        edgecolor="black",
        linewidth=0.8,
        label="Inactive, labeled",
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{strategy}: diagnostic view")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_campaign_initialization_comparison(
    X_pool,
    y_pool,
    histories,
    summary,
    target_round,
    output_path,
):
    """
    Compare campaign outcomes across initialization and selection strategies.

    Parameters
    ----------
    histories : dict
        Keys are (initialization_strategy, selection_strategy).
        Values are run_simulation() histories.
    summary : pd.DataFrame
        Summary table from repeated simulations.
    """

    initialization_strategies = ["random", "diverse"]
    selection_strategies = ["random", "greedy", "uncertainty_topk", "uncertainty_diverse", "query_by_committee"]

    fig, axes = plt.subplots(
        nrows=len(selection_strategies),
        ncols=len(initialization_strategies),
        figsize=(10, 12),
        sharex=True,
        sharey=True,
    )

    for row, selection_strategy in enumerate(selection_strategies):
        for col, initialization_strategy in enumerate(initialization_strategies):
            ax = axes[row, col]

            history = histories[(initialization_strategy, selection_strategy)]

            initial_indices, newly_selected_indices = reconstruct_campaign_state(
                history=history,
                target_round=target_round,
            )

            all_indices = np.arange(len(y_pool))

            tested_indices = np.concatenate(
                [initial_indices, newly_selected_indices]
            )

            untested_indices = np.setdiff1d(all_indices, tested_indices)

            newly_active = newly_selected_indices[
                y_pool[newly_selected_indices] == 1
            ]

            newly_inactive = newly_selected_indices[
                y_pool[newly_selected_indices] == 0
            ]

            ax.scatter(
                X_pool[untested_indices, 0],
                X_pool[untested_indices, 1],
                c="lightgray",
                s=12,
                alpha=0.45,
            )

            ax.scatter(
                X_pool[initial_indices, 0],
                X_pool[initial_indices, 1],
                c="blue",
                s=45,
                edgecolor="black",
                linewidth=0.6,
            )

            ax.scatter(
                X_pool[newly_active, 0],
                X_pool[newly_active, 1],
                c="green",
                s=45,
                edgecolor="black",
                linewidth=0.6,
            )

            ax.scatter(
                X_pool[newly_inactive, 0],
                X_pool[newly_inactive, 1],
                c="red",
                s=45,
                edgecolor="black",
                linewidth=0.6,
            )

            metric_row = summary[
                (summary["initialization_strategy"] == initialization_strategy)
                & (summary["strategy"] == selection_strategy)
                & (summary["round"] == target_round)
            ].iloc[0]

            annotation = (
                f"ROC AUC = {metric_row['roc_auc_mean']:.3f} ± "
                f"{metric_row['roc_auc_std']:.3f}\n"
                f"Actives = {metric_row['actives_mean']:.1f}"
            )

            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(f"{initialization_strategy} init")

            if col == 0:
                ax.set_ylabel(f"{selection_strategy}\nFeature 2")

            if row == len(selection_strategies) - 1:
                ax.set_xlabel("Feature 1")

            ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="lightgray", label="Untested"),
        plt.Line2D([], [], marker="o", linestyle="", color="blue", label="Initial"),
        plt.Line2D([], [], marker="o", linestyle="", color="green", label="New active"),
        plt.Line2D([], [], marker="o", linestyle="", color="red", label="New inactive"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        f"Campaign state after {target_round} active-learning rounds",
        y=0.98,
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_activity_embedding(
    embedding,
    y,
    output_path,
    title="Chemical space colored by activity",
):
    """
    Plot all molecules in 2D chemical space colored by true activity.
    """

    active = y == 1
    inactive = y == 0

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        embedding[inactive, 0],
        embedding[inactive, 1],
        c="lightcoral",
        s=25,
        alpha=0.6,
        label="Inactive",
    )

    ax.scatter(
        embedding[active, 0],
        embedding[active, 1],
        c="lightgreen",
        s=25,
        alpha=0.6,
        label="Active",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_campaign_on_embedding(
    embedding,
    y,
    history,
    target_round,
    output_path,
    title,
):
    """
    Overlay an active-learning campaign on a 2D molecular embedding.
    """

    initial_indices, newly_selected_indices = reconstruct_campaign_state(
        history=history,
        target_round=target_round,
    )

    all_indices = np.arange(len(y))
    tested_indices = np.concatenate([initial_indices, newly_selected_indices])
    untested_indices = np.setdiff1d(all_indices, tested_indices)

    newly_active = newly_selected_indices[y[newly_selected_indices] == 1]
    newly_inactive = newly_selected_indices[y[newly_selected_indices] == 0]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        embedding[untested_indices, 0],
        embedding[untested_indices, 1],
        c="lightgray",
        s=20,
        alpha=0.45,
        label="Untested",
    )

    ax.scatter(
        embedding[initial_indices, 0],
        embedding[initial_indices, 1],
        c="blue",
        s=55,
        edgecolor="black",
        linewidth=0.6,
        label="Initial",
    )

    ax.scatter(
        embedding[newly_active, 0],
        embedding[newly_active, 1],
        c="green",
        s=55,
        edgecolor="black",
        linewidth=0.6,
        label="New active",
    )

    ax.scatter(
        embedding[newly_inactive, 0],
        embedding[newly_inactive, 1],
        c="red",
        s=55,
        edgecolor="black",
        linewidth=0.6,
        label="New inactive",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_embedding_campaign_comparison(
    embedding,
    y,
    histories,
    summary,
    target_round,
    output_path,
    title="Chemical-space campaign comparison",
):
    """
    Multi-panel campaign comparison on a 2D embedding.

    Rows: selection strategies
    Columns: initialization strategies
    """

    initialization_strategies = ["random", "diverse"]
    selection_strategies = [
        "random",
        "greedy",
        "uncertainty_topk",
        "uncertainty_diverse",
    ]

    fig, axes = plt.subplots(
        nrows=len(selection_strategies),
        ncols=len(initialization_strategies),
        figsize=(10, 14),
        sharex=True,
        sharey=True,
    )

    for row, selection_strategy in enumerate(selection_strategies):
        for col, initialization_strategy in enumerate(initialization_strategies):
            ax = axes[row, col]

            history = histories[(initialization_strategy, selection_strategy)]

            initial_indices, newly_selected_indices = reconstruct_campaign_state(
                history=history,
                target_round=target_round,
            )

            all_indices = np.arange(len(y))
            tested_indices = np.concatenate(
                [initial_indices, newly_selected_indices]
            )
            untested_indices = np.setdiff1d(all_indices, tested_indices)

            newly_active = newly_selected_indices[
                y[newly_selected_indices] == 1
            ]
            newly_inactive = newly_selected_indices[
                y[newly_selected_indices] == 0
            ]

            ax.scatter(
                embedding[untested_indices, 0],
                embedding[untested_indices, 1],
                c="lightgray",
                s=14,
                alpha=0.30,
            )

            ax.scatter(
                embedding[initial_indices, 0],
                embedding[initial_indices, 1],
                c="blue",
                s=45,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.scatter(
                embedding[newly_active, 0],
                embedding[newly_active, 1],
                c="green",
                s=45,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.scatter(
                embedding[newly_inactive, 0],
                embedding[newly_inactive, 1],
                c="red",
                s=45,
                edgecolor="black",
                linewidth=0.5,
            )

            metric_row = summary[
                (summary["initialization_strategy"] == initialization_strategy)
                & (summary["strategy"] == selection_strategy)
                & (summary["round"] == target_round)
            ].iloc[0]

            annotation = (
                f"ROC AUC = {metric_row['roc_auc_mean']:.3f} ± "
                f"{metric_row['roc_auc_std']:.3f}\n"
                f"Actives = {metric_row['actives_mean']:.1f}"
            )

            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

            if row == 0:
                ax.set_title(f"{initialization_strategy} init")

            if col == 0:
                ax.set_ylabel(f"{selection_strategy}\nPC2")

            if row == len(selection_strategies) - 1:
                ax.set_xlabel("PC1")

            ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="lightgray", label="Untested"),
        plt.Line2D([], [], marker="o", linestyle="", color="blue", label="Initial"),
        plt.Line2D([], [], marker="o", linestyle="", color="green", label="New active"),
        plt.Line2D([], [], marker="o", linestyle="", color="red", label="New inactive"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        title,
        y=0.985,
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_chemical_space_explanatory_panel(
    pca_embedding,
    pca,
    umap_embedding,
    y,
    output_path,
    title="BindingDB EGFR chemical-space overview",
):
    """
    Explanatory panel for molecular chemical space.

    Includes:
    - PC1 vs PC2
    - PC1 vs PC3
    - PC2 vs PC3
    - UMAP projection
    - PCA cumulative explained variance
    """

    import numpy as np
    import matplotlib.pyplot as plt

    active = y == 1
    inactive = y == 0

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 9),
    )

    projections = [
        (0, 1, "PC1", "PC2"),
        (0, 2, "PC1", "PC3"),
        (1, 2, "PC2", "PC3"),
    ]

    for ax, (i, j, xlabel, ylabel) in zip(axes[0], projections):
        ax.scatter(
            pca_embedding[inactive, i],
            pca_embedding[inactive, j],
            c="lightcoral",
            s=25,
            alpha=0.30,
            label="Inactive",
        )

        ax.scatter(
            pca_embedding[active, i],
            pca_embedding[active, j],
            c="lightgreen",
            s=25,
            alpha=0.30,
            label="Active",
        )

        ax.set_xlabel(f"{xlabel} ({explained[i]:.1%})")
        ax.set_ylabel(f"{ylabel} ({explained[j]:.1%})")
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(
        umap_embedding[inactive, 0],
        umap_embedding[inactive, 1],
        c="lightcoral",
        s=25,
        alpha=0.30,
        label="Inactive",
    )
    ax.scatter(
        umap_embedding[active, 0],
        umap_embedding[active, 1],
        c="lightgreen",
        s=25,
        alpha=0.30,
        label="Active",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP chemical space")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    components = np.arange(1, len(explained) + 1)

    ax.plot(
        components,
        cumulative,
        marker="o",
    )

    ax.bar(
        components,
        explained,
        alpha=0.4,
        label="Individual variance",
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance")
    ax.set_title("PCA explained variance")
    ax.set_ylim(0, min(1.0, cumulative[-1] + 0.05))
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    ax.axis("off")

    text = (
        "Chemical-space interpretation\n\n"
        "PCA projections:\n"
        "- Linear views of Morgan fingerprint variance\n"
        "- PC1/PC2 show the dominant global structure\n"
        "- PC3 can reveal variation hidden in 2D PCA\n\n"
        "UMAP projection:\n"
        "- Nonlinear neighborhood-preserving map\n"
        "- Uses Jaccard distance for binary fingerprints\n"
        "- Better suited to local chemical neighborhoods\n\n"
        f"Dataset:\n"
        f"- Molecules: {len(y)}\n"
        f"- Active fraction: {y.mean():.3f}\n"
        f"- PCA variance PC1+PC2: {(explained[0] + explained[1]):.1%}\n"
        f"- PCA variance PC1+PC2+PC3: "
        f"{(explained[0] + explained[1] + explained[2]):.1%}"
    )

    ax.text(
        0,
        1,
        text,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_classification_informative_panel(
    embedding_pca,
    pca,
    embedding_umap,
    y,
    histories,
    summary,
    target_round,
    output_path,
    target_name="Target",
    model_name="Logistic Regression",
    fingerprint_name="Morgan radius=2, 2048 bits",
    descriptors=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    initialization_strategies = ["random", "diverse"]
    selection_strategies = [
        "random",
        "greedy",
        "uncertainty_topk",
        "uncertainty_diverse",
    ]

    active = y == 1
    inactive = y == 0

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(18, 16),
        sharex=False,
        sharey=False,
    )

    # Columns 1–2: active-learning campaigns
    for row, selection_strategy in enumerate(selection_strategies):
        for col, initialization_strategy in enumerate(initialization_strategies):
            ax = axes[row, col]

            history = histories[(initialization_strategy, selection_strategy)]

            initial_indices, newly_selected_indices = reconstruct_campaign_state(
                history=history,
                target_round=target_round,
            )

            all_indices = np.arange(len(y))
            tested_indices = np.concatenate(
                [initial_indices, newly_selected_indices]
            )
            untested_indices = np.setdiff1d(all_indices, tested_indices)

            newly_active = newly_selected_indices[y[newly_selected_indices] == 1]
            newly_inactive = newly_selected_indices[y[newly_selected_indices] == 0]

            ax.scatter(
                embedding_pca[untested_indices, 0],
                embedding_pca[untested_indices, 1],
                c="lightgray",
                s=12,
                alpha=0.45,
            )

            ax.scatter(
                embedding_pca[initial_indices, 0],
                embedding_pca[initial_indices, 1],
                c="blue",
                s=35,
                edgecolor="black",
                linewidth=0.4,
            )

            ax.scatter(
                embedding_pca[newly_active, 0],
                embedding_pca[newly_active, 1],
                c="green",
                s=35,
                edgecolor="black",
                linewidth=0.4,
            )

            ax.scatter(
                embedding_pca[newly_inactive, 0],
                embedding_pca[newly_inactive, 1],
                c="red",
                s=35,
                edgecolor="black",
                linewidth=0.4,
            )

            metric_row = summary[
                (summary["initialization_strategy"] == initialization_strategy)
                & (summary["strategy"] == selection_strategy)
                & (summary["round"] == target_round)
            ].iloc[0]

            ax.text(
                0.03,
                0.97,
                (
                    f"ROC AUC = {metric_row['roc_auc_mean']:.3f} ± "
                    f"{metric_row['roc_auc_std']:.3f}\n"
                    f"Actives = {metric_row['actives_mean']:.1f}"
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

            if row == 0:
                ax.set_title(f"{initialization_strategy} initialization")

            if col == 0:
                ax.set_ylabel(f"{selection_strategy}\nPC2")

            ax.set_xlabel("PC1")
            ax.grid(alpha=0.25)

    # Column 3: PCA views
    pca_views = [
        (0, 1, "PC1", "PC2"),
        (0, 2, "PC1", "PC3"),
        (1, 2, "PC2", "PC3"),
    ]

    for row, (i, j, xlabel, ylabel) in enumerate(pca_views):
        ax = axes[row, 2]

        ax.scatter(
            embedding_pca[inactive, i],
            embedding_pca[inactive, j],
            c="lightcoral",
            s=18,
            alpha=0.30,
            label="Inactive",
        )

        ax.scatter(
            embedding_pca[active, i],
            embedding_pca[active, j],
            c="lightgreen",
            s=18,
            alpha=0.30,
            label="Active",
        )

        ax.set_xlabel(f"{xlabel} ({explained[i]:.1%})")
        ax.set_ylabel(f"{ylabel} ({explained[j]:.1%})")
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.grid(alpha=0.25)

    # Column 3, row 4: PCA cumulative variance
    ax = axes[3, 2]
    components = np.arange(1, len(explained) + 1)

    ax.bar(
        components,
        explained,
        alpha=0.4,
        label="Individual",
    )

    ax.plot(
        components,
        cumulative,
        marker="o",
        label="Cumulative",
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance")
    ax.set_title("PCA variance")
    ax.set_ylim(0, min(1.0, cumulative[-1] + 0.05))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Column 4, row 1: UMAP
    ax = axes[0, 3]

    ax.scatter(
        embedding_umap[inactive, 0],
        embedding_umap[inactive, 1],
        c="lightcoral",
        s=18,
        alpha=0.30,
        label="Inactive",
    )

    ax.scatter(
        embedding_umap[active, 0],
        embedding_umap[active, 1],
        c="lightgreen",
        s=18,
        alpha=0.30,
        label="Active",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP chemical space")
    ax.grid(alpha=0.25)

    # Column 4, row 2: descriptors
    ax = axes[1, 3]

    if descriptors is not None:
        active = y == 1
        inactive = y == 0

        ax.scatter(
            descriptors.loc[inactive, "MolWt"],
            descriptors.loc[inactive, "TPSA"],
            c="lightcoral",
            s=18,
            alpha=0.30,
            label="Inactive",
        )

        ax.scatter(
            descriptors.loc[active, "MolWt"],
            descriptors.loc[active, "TPSA"],
            c="lightgreen",
            s=18,
            alpha=0.30,
            label="Active",
        )

        ax.set_xlabel("Molecular weight")
        ax.set_ylabel("TPSA")
        ax.set_title("MolWt vs TPSA")
        ax.grid(alpha=0.25)
    else:
        ax.axis("off")

    # Column 4, row 3: legend
    ax = axes[2, 3]
    ax.axis("off")

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="lightgray", label="Untested"),
        plt.Line2D([], [], marker="o", linestyle="", color="blue", label="Initial labeled"),
        plt.Line2D([], [], marker="o", linestyle="", color="green", label="New active"),
        plt.Line2D([], [], marker="o", linestyle="", color="red", label="New inactive"),
        plt.Line2D([], [], marker="o", linestyle="", color="lightgreen", label="Known active"),
        plt.Line2D([], [], marker="o", linestyle="", color="lightcoral", label="Known inactive"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        fontsize=11,
    )

    # Column 4, rows 4: text summary
    for row in [3, 3]:
        axes[row, 3].axis("off")

    best_row = (
        summary[summary["round"] == target_round]
        .sort_values("roc_auc_mean", ascending=False)
        .iloc[0]
    )

    info_text = (
        f"Dataset and experiment\n\n"
        f"Target: {target_name}\n"
        f"Molecules: {len(y)}\n"
        f"Active fraction: {y.mean():.3f}\n"
        f"Features: {fingerprint_name}\n"
        f"Model: {model_name}\n\n"
        f"Active-learning setup\n"
        f"Initial labels: 20\n"
        f"Batch size: 10\n"
        f"Rounds: {target_round}\n\n"
        f"Best strategy at round {target_round}\n"
        f"{best_row['initialization_strategy']} init + "
        f"{best_row['strategy']}\n"
        f"ROC AUC = {best_row['roc_auc_mean']:.3f} ± "
        f"{best_row['roc_auc_std']:.3f}\n\n"
        f"PCA variance\n"
        f"PC1+PC2: {(explained[0] + explained[1]):.1%}\n"
        f"PC1+PC2+PC3: "
        f"{(explained[0] + explained[1] + explained[2]):.1%}"
    )

    axes[3, 3].text(
        0,
        1,
        info_text,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.suptitle(
        f"{target_name}: classification active-learning overview",
        fontsize=17,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_descriptor_space_on_axis(
    ax,
    descriptors,
    y,
    x_col="MolWt",
    y_col="TPSA",
):
    active = y == 1
    inactive = y == 0

    ax.scatter(
        descriptors.loc[inactive, x_col],
        descriptors.loc[inactive, y_col],
        c="lightcoral",
        s=18,
        alpha=0.30,
        label="Inactive",
    )

    ax.scatter(
        descriptors.loc[active, x_col],
        descriptors.loc[active, y_col],
        c="lightgreen",
        s=18,
        alpha=0.30,
        label="Active",
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    ax.grid(alpha=0.25)

    xmax = descriptors["MolWt"].quantile(0.99)
    ymax = descriptors["TPSA"].quantile(0.99)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)


def plot_qbc_diagnostics(
    embedding_pca,
    embedding_umap,
    y,
    disagreement_scores,
    selected_indices,
    output_path,
    title="Query by Committee diagnostics",
):
    import matplotlib.pyplot as plt
    import numpy as np

    active = y == 1
    inactive = y == 0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 9),
    )

    # PCA disagreement map
    ax = axes[0, 0]
    sc = ax.scatter(
        embedding_pca[:, 0],
        embedding_pca[:, 1],
        c=disagreement_scores,
        s=25,
        alpha=0.75,
    )
    ax.set_title("PCA colored by committee disagreement")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Disagreement variance")

    # UMAP disagreement map
    ax = axes[0, 1]
    sc = ax.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=disagreement_scores,
        s=25,
        alpha=0.75,
    )
    ax.set_title("UMAP colored by committee disagreement")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Disagreement variance")

    # Histogram
    ax = axes[1, 0]
    ax.hist(disagreement_scores, bins=30, alpha=0.8)
    ax.set_title("Distribution of committee disagreement")
    ax.set_xlabel("Disagreement variance")
    ax.set_ylabel("Number of molecules")
    ax.grid(alpha=0.25)

    # QBC selected overlay
    ax = axes[1, 1]
    ax.scatter(
        embedding_pca[inactive, 0],
        embedding_pca[inactive, 1],
        c="lightcoral",
        s=20,
        alpha=0.45,
        label="Inactive",
    )
    ax.scatter(
        embedding_pca[active, 0],
        embedding_pca[active, 1],
        c="lightgreen",
        s=20,
        alpha=0.45,
        label="Active",
    )
    ax.scatter(
        embedding_pca[selected_indices, 0],
        embedding_pca[selected_indices, 1],
        c="black",
        s=70,
        marker="x",
        linewidth=1.5,
        label="QBC selected",
    )
    ax.set_title("Top QBC-selected molecules")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)