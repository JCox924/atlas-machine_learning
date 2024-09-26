#!/usr/bin/env python3
"""Module contains class DeepNeuralNetwork with save and load methods"""
import numpy as np
import matplotlib.pyplot as plt
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
        self.cache['A0'] = X
        # Perform forward propagation
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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Y: numpy.ndarray of shape (1, m) containing the correct labels
        A1: numpy.ndarray of shape (nodes, m)
         containing the hidden layer's activated output
        A2: numpy.ndarray of shape (1, m)
         containing the output layer's activated output
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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network
        X: numpy.ndarray of shape (nx, m) containing the input data
        Y: numpy.ndarray of shape (1, m) containing the correct labels
        iterations: number of iterations to train over
        alpha: learning rate
        verbose: if True, prints cost every step iterations
        graph: if True, plots the cost after training
        step: steps for printing and graphing
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            if i % step == 0 or i == 0 or i == iterations:
                cost = self.cost(Y, A2)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not isinstance(filename, str):
            return
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
