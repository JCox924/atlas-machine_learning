#!/usr/bin/env python3
"""
Module for GRUCell implementation.

This module contains the GRUCell class that represents a gated recurrent unit.
It performs forward propagation for one time step using update and reset gates,
and computes the output using a softmax activation.
"""

import numpy as np


def sigmoid(x):
    """
    Computes the sigmoid activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell.

    Attributes:
        Wz (numpy.ndarray): Weights for the update gate.
        Wr (numpy.ndarray): Weights for the reset gate.
        Wh (numpy.ndarray): Weights for the candidate hidden state.
        Wy (numpy.ndarray): Weights for the output.
        bz (numpy.ndarray): Bias for the update gate.
        br (numpy.ndarray): Bias for the reset gate.
        bh (numpy.ndarray): Bias for the candidate hidden state.
        by (numpy.ndarray): Bias for the output.
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRUCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(h + i, h))
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

        z = sigmoid(np.dot(concat, self.Wz) + self.bz)

        r = sigmoid(np.dot(concat, self.Wr) + self.br)

        concat_candidate = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_candidate, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y_linear = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
