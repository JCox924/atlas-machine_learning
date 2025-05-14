#!/usr/bin/env python3
"""
Module for LSTMCell implementation.

This module contains the LSTMCell class that represents an LSTM unit.
It performs forward propagation for one time step using the forget, update,
candidate, and output gates, and produces the output using a softmax activation.
"""

import numpy as np


def sigmoid(x):
    """
    Computes the sigmoid activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying sigmoid.
    """
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    """
    Represents an LSTM unit.

    Attributes:
        Wf (numpy.ndarray): Weights for the forget gate.
        Wu (numpy.ndarray): Weights for the update gate.
        Wc (numpy.ndarray): Weights for the candidate cell state.
        Wo (numpy.ndarray): Weights for the output gate.
        Wy (numpy.ndarray): Weights for the outputs.
        bf (numpy.ndarray): Biases for the forget gate.
        bu (numpy.ndarray): Biases for the update gate.
        bc (numpy.ndarray): Biases for the candidate cell state.
        bo (numpy.ndarray): Biases for the output gate.
        by (numpy.ndarray): Biases for the outputs.
    """

    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            c_prev (numpy.ndarray): Previous cell state of shape (m, h).
            x_t (numpy.ndarray): Data input for the cell of shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state.
            c_next (numpy.ndarray): Next cell state.
            y (numpy.ndarray): Output of the cell using softmax activation.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = sigmoid(np.dot(concat, self.Wf) + self.bf)

        u = sigmoid(np.dot(concat, self.Wu) + self.bu)

        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)

        c_next = f * c_prev + u * c_tilde

        o = sigmoid(np.dot(concat, self.Wo) + self.bo)

        h_next = o * np.tanh(c_next)

        z = np.dot(h_next, self.Wy) + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, c_next, y
