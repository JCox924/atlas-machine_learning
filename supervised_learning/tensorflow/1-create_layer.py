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
    initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_avg')

    weights = tf.get_variable("weights", shape=[prev.get_shape().as_list()[1], n], initializer=initializer)
    biases = tf.get_variable("biases", shape=[n], initializer=tf.zeros_initializer())

    layer = tf.matmul(prev, weights) + biases

    if activation is not None:
        layer = activation(layer)

    return layer

