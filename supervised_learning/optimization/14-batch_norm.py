#!/usr/bin/env python3
"""
Module 14-batch_norm contains functions:
    create_batch_norm(prev, n, activation)
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: activation function to be used on output layer
    Returns:
        tf.keras.layers.BatchNormalization: tensor of the activated output for the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    d_layer = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    batch_norm_layer = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer=tf.keras.initializers.Zeros(),
        gamma_initializer=tf.keras.initializers.Ones()
    )(d_layer)

    if activation is not None:
        output = activation(batch_norm_layer)
    else:
        output = batch_norm_layer
    return output
