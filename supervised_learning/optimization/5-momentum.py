#!/usr/bin/env python3
"""Module momentum contains functions: update_variables_momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.

    Args:
        alpha: The learning rate.
        beta1: The momentum hyperparameter.
        var: The variable to be updated.
        grad: The gradient of the cost with respect to var.
        v: The velocity from the previous iteration.

    Returns:
        var: The updated variable.
        v: The updated velocity.
    """
    v = beta1 * v + (1 - beta1) * grad

    var = var - alpha * v

    return var, v
