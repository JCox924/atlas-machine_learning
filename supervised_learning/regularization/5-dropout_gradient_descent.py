#!/usr/bin/env python3
"""
Module 5-dropout_gradient_descent contains functions:
    dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L)
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with
        Dropout regularization using gradient descent.

    Args:
        Y: a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels
        weights: a dictionary of the weights and biases of the neural network
        cache: a dictionary of the outputs and dropout masks of each layer
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network

    Returns:
        Updates the weights dictionary in place (no return).
    """
    m = Y.shape[1]
    Wb = weights
    A_cur = cache["A" + str(L)]
    dZ = (A_cur - Y)

    for i in range(L, 0, -1):
        W_cur = Wb["W" + str(i)]
        A_cur = cache["A" + str(i)]
        A_prev = cache["A" + str(i - 1)]
        b_cur = Wb["b" + str(i)]

        dW1 = ((1 / m) * np.dot(dZ, A_prev.T))
        db1 = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dg = ((1 - A_prev ** 2))
        dZ = (np.dot(W_cur.T, dZ)) * dg

        if i > 1:
            dZ = dZ * cache["D" + str(i - 1)]
            dZ = dZ / keep_prob

        Wb["W" + str(i)] = W_cur - (alpha * dW1)
        Wb["b" + str(i)] = b_cur - (alpha * db1)

    weights = Wb
