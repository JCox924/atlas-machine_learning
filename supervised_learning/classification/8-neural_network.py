#!/usr/bin/env python3
"""NeuralNetwork class definition"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor.

        Parameters:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer
        self.b1 = np.zeros((nodes, 1))        # Biases for the hidden layer
        self.A1 = 0                           # Activated output for the hidden layer

        self.W2 = np.random.randn(1, nodes)   # Weights for the output neuron
        self.b2 = 0                           # Bias for the output neuron
        self.A2 = 0                           # Activated output (prediction)
