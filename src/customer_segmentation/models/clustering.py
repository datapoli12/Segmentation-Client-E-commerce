"""
Clustering algorithms wrapper.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from typing import Tuple, Optional


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10
) -> Tuple[np.ndarray, KMeans]:
    """
    Run K-means clustering.

    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
        n_init: Number of initializations

    Returns:
        Tuple[labels, model]: Cluster labels and fitted model
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )
    labels = model.fit_predict(X)
    return labels, model


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward"
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    """
    Run Hierarchical Agglomerative Clustering.

    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')

    Returns:
        Tuple[labels, model]: Cluster labels and fitted model
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X)
    return labels, model


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Run DBSCAN clustering.

    Args:
        X: Feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum samples in neighborhood

    Returns:
        Tuple[labels, model]: Cluster labels (-1 = noise) and fitted model
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model
