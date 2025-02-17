#!/usr/bin/env python3
"""
Module that defines the BayesianOptimization class for
optimizing a black-box function
using a noiseless 1D Gaussian process.
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initializes Bayesian optimization.

        Arguments:
            f (callable): The black-box function to be optimized.

            X_init (numpy.ndarray): represents the inputs already sampled.

            Y_init (numpy.ndarray): represents the outputs of the black-box
                function for each input.

            bounds (tuple): represents the bounds of the space in which to
                look for the optimal point.

            ac_samples (int): The number of samples that should be analyzed
                during acquisition.

            l (float): The length parameter for the kernel.

            sigma_f (float): The standard deviation given to the output of
                the black-box function.

            xsi (float): The exploration-exploitation factor for acquisition.

            minimize (bool): True if performing minimization; False if
                performing maximization.

        Public instance attributes:
            f: The black-box function.

            gp: An instance of the GaussianProcess class.

            X_s (numpy.ndarray): contains all acquisition sample points,
                evenly spaced between the specified bounds.

            xsi: The exploration-exploitation factor.

            minimize: Boolean indicating whether
                to minimize (True) or maximize (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)
