import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


def compute_pca_embedding(X, n_components=10, random_state=42):
    """
    Compute PCA embedding with multiple components.
    """

    pca = PCA(
        n_components=n_components,
        random_state=random_state,
    )

    embedding = pca.fit_transform(X)

    return embedding, pca


def compute_umap_embedding(
    X,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="jaccard",
    random_state=42,
):
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        init="random",
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X)

    return embedding, reducer


def pca_variance_table(pca):
    """
    Return explained and cumulative variance for PCA components.
    """

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    return explained, cumulative