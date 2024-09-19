#!/usr/bin/env python3
"""Module contains class DeepNeuralNetwork with save and load methods"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network class"""

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
        layers_arr = np.array(layers)
        if (not issubclass(layers_arr.dtype.type, np.integer)
                or np.any(layers_arr <= 0)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Only one loop allowed: initializing weights
        for i in range(1, self.L + 1):
            layer_size = layers[i - 1]
            if i == 1:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[i - 2]

            self.weights['W' + str(i)] = (
                np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            )
            self.weights['b' + str(i)] = np.zeros((layer_size, 1))

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        """Derivative of the sigmoid activation function"""
        return A * (1 - A)

    def forward_prop(self, X):
        """Forward propagation without loops (unrolled for specific layers)"""
        self.cache['A0'] = X

        # Manually unrolling for layers (adjust if you have more layers)
        A_prev = X
        for i in range(1, self.L + 1):
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            Z = np.dot(W, A_prev) + b
            A = self.sigmoid(Z)
            self.cache['A' + str(i)] = A
            A_prev = A

        return A, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression
        Y: numpy.ndarray of shape (1, m) with correct labels
        A: numpy.ndarray of shape (1, m) with activated output of the neuron
        Returns the cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
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

    def gradient_descent(self, Y, alpha=0.05):
        """Backward propagation without loops (unrolled for specific layers)"""
        m = Y.shape[1]
        weights = self.weights
        cache = self.cache
        L = self.L

        A_final = cache['A' + str(L)]
        dZ = A_final - Y

        for i in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(i - 1)]
            W = weights['W' + str(i)]
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            weights['W' + str(i)] -= alpha * dW
            weights['b' + str(i)] -= alpha * db

            if i > 1:
                A_prev_prev = cache['A' + str(i - 1)]
                dZ = np.dot(W.T, dZ) * self.sigmoid_derivative(A_prev_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Y: numpy.ndarray of shape (1, m) containing the correct labels
        iterations: number of iterations to train over
        alpha: learning rate
        Returns the evaluation of the training data
         after iterations of training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (float, int)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A, _ = self.forward_prop(X)
            self.gradient_descent(Y, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        if not isinstance(filename, str):
            return None
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception:
            pass

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not isinstance(filename, str):
            return None
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                if isinstance(obj, DeepNeuralNetwork):
                    return obj
                else:
                    return None
        except Exception:
            return None
