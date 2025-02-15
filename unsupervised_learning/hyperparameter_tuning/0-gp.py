#!/usr/bin/env python3
"""
Module 0-gp contains:
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
        self.K = self.kernel(X_init, Y_init)

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

        return (self.sigma_f ** 2) * np.exp(( -0.5 * sqdist) / (self.l ** 2))
