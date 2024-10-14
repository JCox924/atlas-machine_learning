#!/usr/bin/env python3
"""
Module l2_reg_create_layer contains functions:
    l2_reg_create_layer(prev, n, activation, lambtha)
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow
        that includes L2 regularization.

    Args:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used on the layer
        lambtha: the L2 regularization parameter

    Returns:
        tf.Tensor: The output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg")
                                                        )
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer
                                  )
    return layer(prev)
