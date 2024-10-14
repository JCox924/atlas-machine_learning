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
    l2_loss = tf.add_n([tf.nn.l2_loss(var)
                        for var in model.trainable_weights
                        if 'kernel' in var.name])
    return cost + l2_loss
