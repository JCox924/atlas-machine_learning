#!/usr/bin/env python3
"""
Module contains:
    functions:
        learning_rate_decay(alpha, decay_rate, decay_step)
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Sets up the learning rate decay operation

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine
            the rate at which alpha will decay
        decay_step: the number of passes of gradient descent
            that should occur before alpha is decayed further
    Returns:
        the learning rate decay operation
            using TensorFlow's InverseTimeDecay function
    """

    operation = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return operation
