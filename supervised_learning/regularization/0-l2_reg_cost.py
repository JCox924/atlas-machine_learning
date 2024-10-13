#!/usr/bin/env python3
"""
Module l2_regularization contains:
    functions:
        l2_reg_cost(cost, lambtha, weights, L, m)
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: dictionary of weights and biases of the neural network
        L: the number of layers in the neural network
        m: the number of data points used

    Returns:
        The cost of the network accounting for L2 regularization
    """
    l2_cost = 0
    for i in range(1, L + 1):
        l2_cost += np.sum(np.square(weights['W' + str(i)]))
    l2_cost *= lambtha / (2 * m)
    return cost + l2_cost
