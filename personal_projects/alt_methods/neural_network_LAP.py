#!/usr/bin/env python3
"""Module contains class NeuralNetwork"""

import numpy as np
import matplotlib.pyplot as plt
import time


class NeuralNetwork:
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

        # velocities for momentum terms

        self.__VdW1 = np.zeros_like(self.__W1)
        self.__Vdb1 = np.zeros_like(self.__b1)
        self.__VdW2 = np.zeros_like(self.__W2)
        self.__Vdb2 = np.zeros_like(self.__b2)

        self.training_time = 0

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

    def least_action_optimizer(self, X, Y, A1, A2, alpha=0.05, gamma=0.9, mass=1.0):
        """
        Performs one pass of an optimizer inspired by the least action principle.
        """
        m = Y.shape[1]

        # Compute gradients (Potential Energy derivative)
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.__W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update momentum (p) variables
        self.__pW2 = gamma * self.__pW2 - alpha * dW2
        self.__pb2 = gamma * self.__pb2 - alpha * db2
        self.__pW1 = gamma * self.__pW1 - alpha * dW1
        self.__pb1 = gamma * self.__pb1 - alpha * db1

        # Update parameters using momentum (akin to position update in physics)
        self.__W2 += self.__pW2 / mass
        self.__b2 += self.__pb2 / mass
        self.__W1 += self.__pW1 / mass
        self.__b1 += self.__pb1 / mass

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.__W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train_LAP(self, X, Y, X_val=None, Y_val=None, iterations=5000, alpha=0.05, gamma=0.9,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network using the least action optimizer
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(gamma, float):
            raise TypeError("gamma must be a float")
        if not 0 < gamma < 1:
            raise ValueError("gamma must be between 0 and 1")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        val_costs = []
        iteration_steps = []

        start_time = time.time()

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.least_action_optimizer(X, Y, A1, A2, alpha, gamma)

            if i % step == 0 or i % 25 == 0 or i == iterations:
                cost = self.cost(Y, A2)
                costs.append(cost)
                iteration_steps.append(i)

                if X_val is not None and Y_val is not None:
                    A1_val, A2_val = self.forward_prop(X_val)
                    val_cost = self.cost(Y_val, A2_val)
                    val_costs.append(val_cost)
                    if verbose:
                        print(f"Iteration: {i}, Cost: {cost}, Validation Cost: {val_cost}")
                else:
                    if verbose:
                        print(f"Iteration: {i}, Cost: {cost}")

        end_time = time.time()
        self.training_time = end_time - start_time

        print(f"Training time: {self.training_time:.2f} seconds using least_action optimizer")

        self.iteration_steps = iteration_steps
        self.costs = costs
        self.val_costs = val_costs

        if graph:
            plt.figure(figsize=(10, 6))
            plt.plot(iteration_steps, costs, 'r-', label='Training Cost')
            if X_val is not None and Y_val is not None:
                plt.plot(iteration_steps, val_costs, 'g-', label='Validation Cost')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training and Validation Cost using LAP Optimizer')
            plt.legend()
            plt.show()

        return self.evaluate(X, Y)

    def train_gradient_descent(self, X, Y, X_val=None, Y_val=None, iterations=5000, alpha=0.05,
                               verbose=True, graph=True, step=100):
        """
        Trains the neural network using standard gradient descent
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
        val_costs = []
        iteration_steps = []

        start_time = time.time()

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A2)
                costs.append(cost)
                iteration_steps.append(i)

                if X_val is not None and Y_val is not None:
                    A1_val, A2_val = self.forward_prop(X_val)
                    val_cost = self.cost(Y_val, A2_val)
                    val_costs.append(val_cost)
                    if verbose:
                        print(f"Iteration {i} iterations: {cost}, Validation cost: {val_cost}")
                    if verbose:
                        print(f"Cost after {i} iterations: {cost}")


        end_time = time.time()
        self.training_time = end_time - start_time

        print(f"Training time: {self.training_time:.2f} seconds with gradient descent")

        self.iteration_steps = iteration_steps
        self.costs = costs
        self.val_costs = val_costs

        if graph:
            plt.figure(figsize=(10, 6))
            plt.plot(iteration_steps, costs, 'b-', label='Training Cost')
            if X_val is not None and Y_val is not None:
                plt.plot(iteration_steps, val_costs, 'g-', label='Validation Cost')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training and Validation Cost using Gradient Descent')
            plt.legend()
            plt.show()

        return self.evaluate(X, Y)
