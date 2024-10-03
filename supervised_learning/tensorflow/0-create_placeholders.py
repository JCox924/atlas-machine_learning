#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for a neural network.

    Args:
    nx: int - the number of feature columns in the data (input size)
    classes: int - the number of classes in the classifier (output size)

    Returns:
    x:  placeholder for input data
    y:  placeholder for one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
