#!/usr/bin/env python3
"""
Module 4-dropout_forward_prop contains functions:
    dropout_forward_prop(X, weights, L, keep_prob)
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: a numpy.ndarray of shape (nx, m) containing the input data for the network
        weights: a dictionary of the weights and biases of the neural network
        L: the number of layers in the network
        keep_prob: the probability that a node will be kept

    Returns:
        A dictionary containing the outputs of each layer and the dropout mask
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(A_prev, W) + b

        if i != L:
            A = np.tanh(Z)
            D = np.random.randn(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D
        else:
            t_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = t_exp / np.sum(t_exp, axis=0, keepdims=True)

        cache['A' + str(i)] = A

    return cache