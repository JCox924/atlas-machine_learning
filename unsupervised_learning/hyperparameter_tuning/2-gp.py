#!/usr/bin/env python3
"""
Module 1-gp contains:
    class(es):
        GaussianProcess
"""
import numpy as np


class GaussianProcess:
    """
    represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        constructor initializes Gaussian process
        Arguments:
            X_init (ndarray): array of shape (t, 1) representing the
                inputs already sampled with the black-box function
            Y_init (ndarray): array of shape (t, 1) representing the outputs
            t:  number of initial samples
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        Arguments:
            X1: array of shape (m, 1)
            X2: array of shape (n, 1)
        Returns:
            covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = (X1 - X2.T) ** 2
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        """
        predicts the mean and standard deviation
        of points in a Gaussian process.

        Arguments:
            X_s: the points whose mean and standard
            deviation should be calculated.
        Returns:
            mu, sigma
                mu is a numpy.ndarray shape (s,) containing
                mean of each point in X_s
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov)
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian process with a new sample

        Arguments:
            X_new: represents the new sample point
            Y_new: represents the new sample function value
        -- Updates the public instance attributes X, Y, and K
        """
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)
        self.K = self.kernel(self.X, self.X)
