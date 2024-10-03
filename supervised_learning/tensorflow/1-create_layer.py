#!/usr/bin/env python3
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Args:
    prev: tensor output of the previous layer
    n: number of nodes in the layer to create
    activation: activation function to apply

    Returns:
    tensor output of the created layer
    """
    initializer = tf.variance_scaling_initializer(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=initializer, name='layer')
    return layer(prev)

