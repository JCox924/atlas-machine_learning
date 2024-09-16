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

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.

        Parameters:
            Y (numpy.ndarray): True labels of shape (1, m)
            A (numpy.ndarray): Activated output of shape (1, m)

        Returns:
            float: The cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):

        """
        Evaluate the neuron’s predictions.
        Parameters:
            X (numpy.ndarray): Input labels of shape (nx, m)
            Y (numpy.ndarray): True output of shape (1, m)
        Returns:
            tuple: The neuron's prediction and the cost.
        """

        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_pred = np.where(A >= 0.5, 1, 0)
        return Y_pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):

        """
            Calculates one pass of gradient descent on the neuron.

            Parameters:
                X (numpy.ndarray): Input data of shape (nx, m)
                    - nx: number of input features
                    - m: number of examples
                Y (numpy.ndarray): Correct labels of shape (1, m)
                A (numpy.ndarray): Activated output of the
                    neuron for each example of shape (1, m)
                alpha (float): Learning rate

            Updates:
                __W (numpy.ndarray): The weights of
                    the neuron after gradient descent
                __b (float): The bias of the neuron
                    after gradient descent
            """

        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m)
            Y (numpy.ndarray): Correct labels of shape (1, m)
            iterations (int): Number of iterations to train over
            alpha (float): Learning rate

        Raises:
            TypeError: If iterations is not an integer
            ValueError: If iterations is not positive
            TypeError: If alpha is not a float
            ValueError: If alpha is not positive

        Updates:
            __W (numpy.ndarray): The weights of the neuron after training
            __b (float): The bias of the neuron after training
            __A (numpy.ndarray): The activated output of the neuron after training

        Returns:
            tuple: The evaluation of the training
                data after iterations of training have occurred
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
