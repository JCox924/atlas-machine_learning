#!/usr/bin/env python3
"""
This module contains the class Poisson
"""


class Poisson:

    """
    Represents a poisson distribution

    methods:
    - exp
    - pdf
    - cdf
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution with given data or lambtha.

        :Args:
        - data (list): List of data points to estimate lambtha.
        - lambtha (float): Expected number of
        occurrences in a given time frame.

        :Methods:
        - factorial
        - exp
        - pmf
        - cdf

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If lambtha is not positive or
        if data contains fewer than two points.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """
        A helper method to calculate the factorial of a number without math.
        """
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp(self, x):
        """
        A helper method to calculate an approximation of e^x without math.
        Uses a basic Taylor series expansion for e^x.
        """
        if x == 0:
            return 1
        elif x < 0:
            return 1 / self.exp(-x)

        result = 1
        term = 1
        for i in range(1, 1000):
            term *= x / i
            result += term
        return result

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'.

        Parameters:
        - k: The number of successes.

        Returns:
        - The PMF value for k.
        """
        # Convert k to an integer if it isn't
        k = int(k)

        e = 2.7182818285

        # Check if k is valid (k should be non-negative)
        if k < 0:
            return 0

        # Calculate PMF using the Poisson formula
        pmf_value = \
            (self.lambtha ** k * e ** -self.lambtha) / self.factorial(k)
        return round(pmf_value, 10)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of 'successes'.

        Parameters:
        - k: The number of successes.

        Returns:
        - The CDF value for k.
        """
        k = int(k)

        if k < 0:
            return 0

        cdf_value = 0
        for i in range(0, k + 1):
            cdf_value += self.pmf(i)
        return round(cdf_value, 10)
