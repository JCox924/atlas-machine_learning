#!/usr/bin/env python3

"""This module contains the class Normal"""


class Normal:
    """Represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution with given data, mean, or standard deviation.

        Parameters:
        - data (list, optional): List of data points to estimate mean and stddev.
        - mean (float): Mean of the distribution.
        - stddev (float): Standard deviation of the distribution.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If stddev is not positive or if data contains fewer than two points.
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