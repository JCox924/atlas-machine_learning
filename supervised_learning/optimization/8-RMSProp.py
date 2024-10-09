#!/usr/bin/env python3
import tensorflow as tf
"""
Module contains:
    functions:
        create_RMSProp_op(alpha, beta2, epsilon)
"""


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Arg:
        alpha: the learning rate
        beta2: the RMSProp weight (Discounting factor)
        epsilon: a small number to avoid division by zero
    Returns:
        tf.keras.optimizers.RMSprop: A TensorFlow optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2,
                                            epsilon=epsilon)

    return optimizer
