#!/usr/bin/env python3
"""Module contains class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        Initializes the deep neural network
        nx: int, number of input features
        layers: list of the number of nodes in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        layers_arr = np.array(layers)

        if (not np.issubdtype(layers_arr.dtype, np.integer)
                or np.any(layers_arr <= 0)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.L + 1):
            if i == 1:
                self.__weights['W' + str(i)] = (np.random.randn(layers[i - 1], nx)
                                                * np.sqrt(2 / nx))
            else:
                self.__weights['W' + str(i)] = (
                        np.random.randn(layers[i - 1], layers[i - 2])
                        * np.sqrt(2 / layers[i - 2]))
            self.__weights['b' + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary"""
        return self.__weights
