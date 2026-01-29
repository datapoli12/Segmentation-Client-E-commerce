"""
Clustering evaluation metrics.
"""
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.utils import resample
from typing import Dict, Callable


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute internal clustering validation metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        dict: Metrics dictionary
            - silhouette: [-1, 1], higher is better
            - davies_bouldin: [0, inf), lower is better
            - calinski_harabasz: [0, inf), higher is better
    """
    # Exclude noise points for metrics (DBSCAN)
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters": len(np.unique(labels[labels != -1])),
            "noise_pct": (labels == -1).mean() * 100
        }

    X_valid = X[mask]
    labels_valid = labels[mask]

    return {
        "silhouette": silhouette_score(X_valid, labels_valid),
        "davies_bouldin": davies_bouldin_score(X_valid, labels_valid),
        "calinski_harabasz": calinski_harabasz_score(X_valid, labels_valid),
        "n_clusters": len(np.unique(labels_valid)),
        "noise_pct": (labels == -1).mean() * 100
    }


def stability_score(
    X: np.ndarray,
    clustering_func: Callable,
    n_bootstrap: int = 10,
    sample_ratio: float = 0.8,
    random_state: int = 42
) -> float:
    """
    Compute clustering stability via bootstrap resampling.

    Args:
        X: Feature matrix
        clustering_func: Function that takes X and returns labels
        n_bootstrap: Number of bootstrap iterations
        sample_ratio: Proportion of data to sample
        random_state: Random seed

    Returns:
        float: Mean Adjusted Rand Index between bootstrap runs
    """
    np.random.seed(random_state)
    n_samples = int(len(X) * sample_ratio)

    all_labels = []
    indices_list = []

    for i in range(n_bootstrap):
        idx = np.random.choice(len(X), size=n_samples, replace=False)
        X_sample = X[idx]
        labels = clustering_func(X_sample)
        all_labels.append(labels)
        indices_list.append(idx)

    # Compare consecutive bootstrap runs on overlapping samples
    ari_scores = []
    for i in range(n_bootstrap - 1):
        common = np.intersect1d(indices_list[i], indices_list[i + 1])
        if len(common) > 10:
            mask_i = np.isin(indices_list[i], common)
            mask_j = np.isin(indices_list[i + 1], common)
            ari = adjusted_rand_score(
                all_labels[i][mask_i],
                all_labels[i + 1][mask_j]
            )
            ari_scores.append(ari)

    return np.mean(ari_scores) if ari_scores else np.nan
