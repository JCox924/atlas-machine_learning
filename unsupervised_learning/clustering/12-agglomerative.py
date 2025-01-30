#!/usr/bin/env python3
"""
Performs agglomerative clustering on a dataset and displays a dendrogram.
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset using Ward linkage
    and displays the corresponding dendrogram.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        dist (float): the maximum cophenetic distance for all clusters

    Returns:
        clss (numpy.ndarray): shape (n,) containing the cluster indices
                              for each data point
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
