#!/usr/bin/env python3

"""This module contains the class Binomial"""


class Binomial:
    """This class represents the binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution with given data, n, or p.

        Parameters:
        - data (list, optional): List of data points to estimate n and p.
        - n (int): Number of Bernoulli trials.
        - p (float): Probability of success in each trial.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If n is not a positive value or if p is not between 0 and 1.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)

            self.p = mean / self.n

    def factorial(self, x):
        """
        Helper method to calculate the factorial of a number.
        """
        if x == 0 or x == 1:
            return 1
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'.

        Parameters:
        - k (int): The number of successes.

        Returns:
        - float: The PMF value for k.
        """
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        nCk = self.factorial(self.n) / (self.factorial(k)
                                        * self.factorial(self.n - k))

        pmf_value = nCk * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_value
