#!/usr/bin/env python3
"""
Calculates a GMM from a dataset using sklearn.mixture.GaussianMixture.
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): the number of clusters

    Returns:
        pi, m, S, clss, bic
            pi is a numpy.ndarray of shape (k,) containing the cluster priors
            m is a numpy.ndarray of shape (k, d) containing centroid means
            S is a numpy.ndarray of shape (k, d, d) containing the covariance
            clss is a numpy.ndarray of shape (n,) containing the cluster
                 indices for each data point
            bic is a numpy.ndarray with one element for the single value of k
                 (i.e., shape (1,)) containing the BIC value
    """
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
