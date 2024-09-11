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
        - ValueError: If stddev is not positive or
            if data contains fewer than two points.
        """
        if data is None:
            # Use given mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and standard deviation from data
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

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

    def cdf(self, x):
        """
        Calculates the CDF value for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The CDF value for x.
        """
        # CDF uses the error function approximation
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        return 0.5 * (1 + self.erf(z))

    def erf(self, z):
        """
        Approximate the error function (erf) for the CDF calculation using the Abramowitz and Stegun approximation.

        Parameters:
        - z (float): The z-score.

        Returns:
        - float: The approximate value of erf(z).
        """
        # Abramowitz and Stegun approximation constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # Save the sign of z
        sign = 1 if z >= 0 else -1
        z = abs(z)

        # Abramowitz and Stegun formula
        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * self.exp(-z * z)

        return sign * y

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Parameters:
        - x (float): The x-value.

        Returns:
        - float: The PDF value for x.
        """
        # Approximate value of pi
        pi = 3.141592653589793

        # Calculate exponent part
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2

        # Return the PDF value
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * self.exp(exponent)

    def exp(self, x):
        """
        Helper method to calculate the exponential of x using a Taylor series expansion.

        Parameters:
        - x (float): The exponent.

        Returns:
        - float: An approximation of e^x.
        """
        result = 1
        term = 1
        for i in range(1, 100):
            term *= x / i
            result += term
        return result
