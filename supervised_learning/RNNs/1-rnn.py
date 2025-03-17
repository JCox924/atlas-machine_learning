#!/usr/bin/env python3
"""
Module for performing forward propagation for a simple RNN.

This module contains the rnn function that uses a given RNNCell instance to
perform forward propagation over a sequence of time steps.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: An instance of RNNCell that will be used for the forward propagation.
        X (numpy.ndarray): Data to be used, of shape (t, m, i) where:
                           t is the number of time steps,
                           m is the batch size, and
                           i is the dimensionality of the data.
        h_0 (numpy.ndarray): Initial hidden state, of shape (m, h) where h is the
                             dimensionality of the hidden state.

    Returns:
        H (numpy.ndarray): Array containing all of the hidden states, with shape (t + 1, m, h).
                           The first hidden state is h_0.
        Y (numpy.ndarray): Array containing all of the outputs, with shape (t, m, o),
                           where o is the dimensionality of the outputs.
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.empty((t + 1, m, h))
    Y = np.empty((t, m, o))
    H[0] = h_0

    for time in range(t):
        h_prev = H[time]
        x_t = X[time]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[time + 1] = h_next
        Y[time] = y

    return H, Y
