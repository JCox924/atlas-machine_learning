#!/usr/bin/env python3
"""Module momentum contains functions: create_momentum(alpha, beta1)"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the gradient descent operation using momentum in TensorFlow.

    Args:
        alpha: The learning rate
        beta1: The momentum hyperparameter

    Returns:
        tf.keras.optimizers.SGD: A TensorFlow optimizer
        that applies gradient descent with momentum.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
