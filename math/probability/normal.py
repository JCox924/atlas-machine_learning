#!/usr/bin/env python3

"""This module contains the class Normal"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution
        with given data, mean, or standard deviation.

        Parameters:
        - data (list, optional): List of data
            points to estimate mean and stddev.
        - mean (float): Mean of the distribution.
        - stddev (float): Standard deviation of the distribution.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If stddev is not positive
            or if data contains fewer than two points.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            self.stddev = (sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The z-score of the x-value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Parameters:
        - z (float): The z-score.

        Returns:
        - float: The x-value corresponding to the z-score.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The PDF value for x.
        """
        pi = 3.1415926536
        e = 2.7182818285
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return (1 + self.erf(z)) / 2

    def erf(self, x):
        """
        Approximate the error function (erf) for the CDF calculation using the given Taylor series expansion.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The approximate value of erf(x).
        """
        pi = 3.1415926536

        term1 = x
        term2 = -(x ** 3) / 3
        term3 = (x ** 5) / 10
        term4 = -(x ** 7) / 42
        term5 = (x ** 9) / 216

        # Sum the series
        erf_approx = (2 / (pi ** 0.5)) * (term1 + term2 + term3 + term4 + term5)

        return erf_approx

    def exp(self, x):
        """
        Approximates the exponential function.

        Parameters:
        - x (float): The exponent.

        Returns:
        - float: An approximation of e^x.
        """

        if x == 0:
            return 1
        elif x < 0:
            return 1 / self.exp(-x)

        n = 1000  # Number of terms in the series
        result = 1.0
        term = 1.0
        for i in range(1, n):
            term *= x / i
            result += term
            if abs(term) < 1e-15:
                break
        return result
