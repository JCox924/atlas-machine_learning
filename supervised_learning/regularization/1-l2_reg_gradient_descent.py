#!/usr/bin/env python3
"""
Module 1-l2_reg_gradient_descent contains functions:
    l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L)
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
        using gradient descent with L2 regularization.

    Args:
        Y: a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data
        weights: dictionary of weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: the learning rate
        lambtha: the L2 regularization parameter
        L: the number of layers of the network

    Returns:
        None
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)] if i > 1 else cache['A0']
        W = weights['W' + str(i)]

        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.sqaure(A_prev))
