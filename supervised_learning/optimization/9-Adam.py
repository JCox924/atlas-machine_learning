#!/usr/bin/env python3
"""
Module 9-Adam contains:
    functions:
        update_variables_Adam(alpha, beta1, beta2, epsilon, var,...)
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, s, v, t):
    """
    Updates the variables of a model using Adam.
    Args:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
        var: the variable to be updated
        grad: the gradient of var
        s: the first moment
        v: the second moment
        t: the number of iterations
    Returns:
        the updated variable, the new first moment,
        and the new second moment
    """
    s = np.add(beta1 * s, (1 - beta1) * grad)

    v = np.add(beta2 * v, (1 - beta2) * np.square(grad))

    m_corrected = np.divide(s, (1 - beta1 ** t))

    v_corrected = np.divide(v, (1 - beta2 ** t))

    var = np.subtract(var, alpha * np.divide(m_corrected, (np.sqrt(v_corrected) + epsilon)))

    return var, s, v
