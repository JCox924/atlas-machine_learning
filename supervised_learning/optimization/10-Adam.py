#!/usr/bin/env python3
import tensorflow as tf
"""
Module contains:
    functions:
        create_Adam_op(alpha, beta1, beta2, epsilon)
"""


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimizer
    Args:
        alpha: the learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: a small number to avoid division by zero
    Returns:
        tf.keras.optimizers.Adam: A TensorFlow optimizer
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,
                                         beta_1=beta1,
                                         beta_2=beta2,
                                         epsilon=epsilon)
    return optimizer
