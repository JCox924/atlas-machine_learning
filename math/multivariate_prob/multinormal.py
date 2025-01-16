#!/usr/bin/env python3
"""
Module multinormal contains:
    class:
        MultiNormal
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.

    Attributes:
        mean : numpy.ndarray
            Shape (d, 1) containing the mean of the data set.
        cov : numpy.ndarray
            Shape (d, d) containing the covariance matrix of the data set.

    Methods:
        __init__(self, data):
            Constructor that initializes and calculates the distribution parameters.
        pdf(self, x):
            Returns the probability density of the distribution.
    """

    def __init__(self, data):
        """
        Constructor for MultiNormal

        Arguments:
            data: numpy.ndarray
                The data set, of shape (d, n) where:
                d is the number of dimensions in each data point
                n is the number of data points

        Raises:
            TypeError:
                If data is not a 2D numpy.ndarray
            ValueError:
                If n < 2
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point x.

        Arguments:
            x: numpy.ndarray
                A data point of shape (d, 1).

        Returns:
            float:
                The value of the PDF at x.

        Raises:
            TypeError:
                If x is not a numpy.ndarray.
            ValueError
                If x is not of shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != self.mean.shape[0]:
            d = self.mean.shape[0]
            raise ValueError(f"x must have the shape ({d}, 1)")

        d = self.mean.shape[0]

        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)

        x_centered = x - self.mean

        # PDF formula:
        # (1 / sqrt((2 pi)^d * det(Sigma))) * exp(-0.5 * (x - mu)^T * Sigma^-1 * (x - mu))
        denom = np.sqrt(((2 * np.pi) ** d) * cov_det)
        exponent = -0.5 * (x_centered.T @ cov_inv @ x_centered)
        pdf_value = (1.0 / denom) * np.exp(exponent)

        return pdf_value[0, 0]
