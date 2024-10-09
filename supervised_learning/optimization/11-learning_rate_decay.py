#!/usr/bin/env python3
"""
Module 11-learning_rate_decay contains:
    functions:
        learning_rate_decay(alpha, decay_rate, global_step, decay_step)
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using exponential decay.
        The new learning rate is given by:
            alpha / (1 + decay_rate * (global_step // decay_step))
    Args:
        alpha: the learning rate
        decay_rate: the weight used to determine
            the rate at which alpha will decay
        global_step: the number of passes of
            gradient descent that have elapsed
        decay_step: the number of passes of
            gradient descent that must elapse before alpha decays
    Returns:
        the updated value for alpha
    """
    new_alpha = alpha / (1 + decay_rate * (global_step // decay_step))

    return new_alpha
