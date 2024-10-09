#!/usr/bin/env python3
"""Module RMSProp contains:
    functions:
        update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s)
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates the variables of a model using RMSProp.

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
        var: a numpy.ndarray containing the variable to be updated
        grad: a numpy.ndarray containing the gradient of var
        s: the previous second moment of var
    Returns:
        the updated variable and the new moment
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var -= alpha * grad / (epsilon + s ** 0.5)
    return var, s
