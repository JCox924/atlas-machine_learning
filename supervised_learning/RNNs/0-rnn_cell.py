#!/usr/bin/env python3
"""
Module for RNNCell implementation.

This module contains the RNNCell class that represents a cell of a simple RNN.
It performs forward propagation for one time step using a tanh activation for
the hidden state and a softmax activation for the output.
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.

    Attributes:
        Wh (numpy.ndarray): Weights for the concatenated hidden state and input data.
        bh (numpy.ndarray): Biases for the hidden state.
        Wy (numpy.ndarray): Weights for the output.
        by (numpy.ndarray): Biases for the output.
    """

    def __init__(self, i, h, o):
        """
        Initializes the RNNCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state with shape (m, h).
            x_t (numpy.ndarray): Data input for the cell with shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state.
            y (numpy.ndarray): Output of the cell using softmax activation.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        z = np.dot(h_next, self.Wy) + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return h_next, y
