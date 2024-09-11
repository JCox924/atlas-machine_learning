#!/usr/bin/env python3


class Exponential:
    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution with given data or lambtha.

        Parameters:
        - data (list, optional): List of data
            points to estimate lambtha.
        - lambtha (float): Expected number of
            occurrences in a given time frame.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If lambtha is not positive or if data contains fewer than two points.
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
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the PDF for a given time period 'x'.

        Parameters:
        - x (float): The time period.

        Returns:
        - float: The PDF value for the time period 'x'.
        """
        if x < 0:
            return 0
        return self.lambtha * self.exp(-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the CDF for a given time period 'x'.

        Parameters:
        - x (float): The time period.

        Returns:
        - float: The CDF value for the time period 'x'.
        """
        if x < 0:
            return 0
        # CDF formula for exponential distribution: 1 - e^(-lambtha * x)
        return 1 - self.exp(-self.lambtha * x)

    def exp(self, x):
        """
        Helper method to approximate the
        exponential of 'x' without using math.exp.

        Parameters:
        - x (float): The exponent.

        Returns:
        - float: An approximation of e^x.
        """
        result = 1
        term = 1
        for i in range(1, 100):  # Using 100 terms for a reasonable approximation
            term *= x / i
            result += term
        return result
