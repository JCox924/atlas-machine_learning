#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Args:
    loss: the loss of the network's prediction
    alpha: the learning rate

    Returns:
    an operation that trains the network using gradient descent
    """

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
