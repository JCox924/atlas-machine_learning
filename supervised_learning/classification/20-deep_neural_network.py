#!/usr/bin/env python3
"""Module contains class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class."""
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
                self.__weights['W' + str(i)] = \
                    (np.random.randn(layers[i - 1], nx) * np.sqrt(2 / nx))
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

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """self propagation"""
        self.cache['A0'] = X
        for i in range(1, self.L + 1):
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            A_prev = self.cache['A' + str(i - 1)]
            Z = np.dot(W, A_prev) + b
            A = self.sigmoid(Z)
            self.cache['A' + str(i)] = A
        return A, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression
        Y: numpy.ndarray of shape (1, m) with correct labels
        A: numpy.ndarray of shape (1, m) with activated output of the neuron
        Returns the cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                 * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        X: numpy.ndarray of shape (nx, m) containing the input data
        Y: numpy.ndarray of shape (1, m) containing the correct labels
        Returns the predictions and cost of the network
        """
        A, _ = self.forward_prop(X)

        prediction = np.where(A >= 0.5, 1, 0)

        cost = self.cost(Y, A)

        return prediction, cost
