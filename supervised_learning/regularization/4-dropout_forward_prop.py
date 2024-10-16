#!/usr/bin/env python3
"""
Module 4-dropout_forward_prop contains functions:
    dropout_forward_prop(X, weights, L, keep_prob)
"""
import numpy as np


def softmax(Z):
    """Softmax activation helper function for stability"""
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


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
    np.random.seed(0)
    cache = {}
    cache['A0'] = X
    A = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        Z = np.matmul(W, A) + b

        if i == L:
            A = softmax(Z)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D
        cache['A' + str(i)] = A
    return cache
