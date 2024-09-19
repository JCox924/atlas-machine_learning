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
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.L + 1):
            if l == 1:
                self.__weights['W' + str(l)] = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(l)] = (
                        np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2 / layers[l - 2]))
            self.__weights['b' + str(l)] = np.zeros((layers[l - 1], 1))

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
        """
        Calculates the forward propagation of the deep neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Updates __cache with the activated outputs
        Returns the output of the neural network and the cache
        """
        self.__cache['A0'] = X  # Save input layer's activations

        for l in range(1, self.__L + 1):
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]
            A_prev = self.__cache['A' + str(l - 1)]

            Z = np.dot(W, A_prev) + b
            A = self.sigmoid(Z)

            self.__cache['A' + str(l)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression
        Y: numpy.ndarray of shape (1, m) with correct labels
        A: numpy.ndarray of shape (1, m) with activated output of the neuron
        Returns the cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
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
        A1: numpy.ndarray of shape (nodes, m) containing the hidden layer's activated output
        A2: numpy.ndarray of shape (1, m) containing the output layer's activated output
        alpha: learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        W2 = self.__weights['W2']
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__weights['W2'] -= alpha * dW2
        self.__weights['b2'] -= alpha * db2
        self.__weights['W1'] -= alpha * dW1
        self.__weights['b1'] -= alpha * db1

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
            A, _ = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__cache['A1'], A, alpha)

        return self.evaluate(X, Y)
