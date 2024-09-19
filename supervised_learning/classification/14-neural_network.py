#!/usr/bin/env python3
"""Module contains class NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """NeuralNetwork class"""
    def __init__(self, nx, nodes):
        """
        Initialize the neural network
        nx: int, number of input features
        nodes: int, number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2"""
        return self.__A2

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Returns the activated outputs A1 and A2
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)

        return self.__A1, self.__A2

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
        A1, A2 = self.forward_prop(X)

        prediction = np.where(A2 >= 0.5, 1, 0)

        cost = self.cost(Y, A2)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Y: numpy.ndarray of shape (1, m) containing the correct labels
        A1: numpy.ndarray of shape (nodes, m) containing the
         hidden layer's activated output
        A2: numpy.ndarray of shape (1, m) containing the
         output layer's activated output
        alpha: learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Y: numpy.ndarray of shape (1, m) containing the correct labels
        iterations: number of iterations to train over
        alpha: learning rate
        Returns the evaluation of the training data after iterations of training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
