#!/usr/bin/env python3
"""
Module 6-dropout_create_layer contains functions:
    dropout_create_layer(prev, n, activation, keep_prob, training=True)
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function for the new layer
        keep_prob: the probability that a node will be kept
        training: a boolean indicating whether the model is in training mode

    Returns:
        The output of the new layer with dropout applied if training.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode="fan_avg"
                                                        )
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    x = dense_layer(prev)
    x = dropout_layer(x, training=training)
    return x
