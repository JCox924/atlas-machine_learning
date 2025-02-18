#!/usr/bin/env python3
"""
Module that defines the BayesianOptimization class for
optimizing a black-box function
using a noiseless 1D Gaussian process.
"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Computes next best sample from the black-box function using the
        Expected Improvement acquisition function.

        Returns:
            X_next (numpy.ndarray): the next best sample from the black-box
                function.
            EI (numpy.ndarray): the expected improvement of
                each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = mu - f_best - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(mu)
            nonzero_sig = sigma > 0
            Z[nonzero_sig] = imp[nonzero_sig] / sigma[nonzero_sig]

        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0] = 0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        Arguments:
            iterations (int): number of iterations to run

        Returns:
            X_opt (numpy.ndarray):  represents the optimal point

            Y_opt (numpy.ndarray): represents the optimal function value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if np.any(np.isclose(self.gp.X.flatten(), X_next, atol=1e-8)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt, Y_opt = self.X_s[idx_opt], self.gp.Y[idx_opt]

        return X_opt, Y_opt
