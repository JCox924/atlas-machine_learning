#!/usr/bin/env python3
"""
Module 5-dropout_gradient_descent contains functions:
    dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L)
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using gradient descent.

    Args:
        Y: a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels
        weights: a dictionary of the weights and biases of the neural network
        cache: a dictionary of the outputs and dropout masks of each layer
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network

    Returns:
        Updates the weights dictionary in place (no return).
    """
    m = Y.shape[1]
    dZ = {}
    for i in reversed(range(1, L + 1)):
        A_curr = cache['A{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        if i == L:
            dZ[i] = A_curr - Y
        else:
            D_curr = cache['D{}'.format(i)]
            dA = np.matmul(weights['W{}'.format(i + 1)].T, dZ[i + 1])
            dA *= D_curr
            dA /= keep_prob
            dZ[i] = dA * (1 - A_curr ** 2)
        dW = (1 / m) * np.matmul(dZ[i], A_prev.T)
        db = (1 / m) * np.sum(dZ[i], axis=1)
        weights['W{}'.format(i)] -= alpha * dW
        weights['b{}'.format(i)] -= alpha * db
