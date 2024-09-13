#!/usr/bin/env python3

"""Neuron class definition"""

import numpy as np


class Neuron:

    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):

        """
        Initialize a Neuron instance.

        Parameters:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)

        self.__b = 0

        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A


    def forward_prop(self, X):

        """
        Perform forward propagation.
        Parameters:
            X (numpy.ndarray): The input data of shape (nx, m)
            where nx is number of features and m is the number of examples.
        Returns:
            numpy.ndarray: The activated output of the neuron (self.__A)
        """

        z = np.dot(self.__W, X) + self.__b

        self.__A = 1 / (1 + np.exp(-z))

        return self.__A
