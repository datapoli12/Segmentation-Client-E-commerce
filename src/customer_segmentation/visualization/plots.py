"""
Visualization functions for clustering analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import List, Optional, Tuple


def plot_elbow(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot elbow curve for K-means.

    Args:
        X: Feature matrix
        k_range: Range of K values to test
        random_state: Random seed
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Elbow plot
    axes[0].plot(list(k_range), inertias, 'bo-')
    axes[0].set_xlabel('Nombre de clusters (K)')
    axes[0].set_ylabel('Inertie (WCSS)')
    axes[0].set_title('Méthode du coude')
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(list(k_range), silhouettes, 'ro-')
    axes[1].set_xlabel('Nombre de clusters (K)')
    axes[1].set_ylabel('Score Silhouette')
    axes[1].set_title('Score Silhouette moyen')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot silhouette diagram for cluster analysis.

    Args:
        X: Feature matrix
        labels: Cluster labels
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(np.unique(labels[labels != -1]))
    silhouette_avg = silhouette_score(X[labels != -1], labels[labels != -1])
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Moyenne: {silhouette_avg:.3f}")
    ax.set_xlabel("Score Silhouette")
    ax.set_ylabel("Cluster")
    ax.set_title("Diagramme Silhouette")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot 2D projections of clusters (pairwise features).

    Args:
        X: Feature matrix (3 features for RFM)
        labels: Cluster labels
        feature_names: Feature names for axes
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if feature_names is None:
        feature_names = ["Recency", "Frequency", "Monetary"]

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (i, j) in zip(axes, pairs):
        scatter = ax.scatter(X[:, i], X[:, j], c=labels, cmap='viridis', alpha=0.6, s=10)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(f"{feature_names[i]} vs {feature_names[j]}")

    plt.colorbar(scatter, ax=axes, label='Cluster')
    plt.tight_layout()
    return fig


def plot_rfm_distributions(
    df: pd.DataFrame,
    columns: List[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot RFM feature distributions.

    Args:
        df: DataFrame with RFM features
        columns: Columns to plot
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if columns is None:
        columns = ["Recency", "Frequency", "Monetary"]

    fig, axes = plt.subplots(1, len(columns), figsize=figsize)

    for ax, col in zip(axes, columns):
        sns.histplot(df[col], ax=ax, kde=True, bins=50)
        ax.set_title(f"Distribution {col}")
        ax.set_xlabel(col)

    plt.tight_layout()
    return fig
