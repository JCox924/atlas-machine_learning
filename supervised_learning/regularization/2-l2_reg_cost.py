#!/usr/bin/env python3
"""
Module 2-l2_reg_cost contains functions:
    l2_reg_keras_cost(cost, model)
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with
        L2 regularization using a Keras model.

    Args:
        cost: a tensor containing the cost of
            the network without L2 regularization
        model: a Keras model that includes layers with L2 regularization

    Returns:
        tf.Tensor: A tensor containing the total cost
            for each layer of the network,
            accounting for L2 regularization
    """
    l2_losses = []

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
            l2_loss = tf.reduce_sum(layer.losses)
            l2_losses.append(l2_loss)

    l2_total_cost = tf.add_n(l2_losses)

    total_cost = cost + l2_total_cost

    return tf.stack([l2_losses[0], l2_losses[1], total_cost])
