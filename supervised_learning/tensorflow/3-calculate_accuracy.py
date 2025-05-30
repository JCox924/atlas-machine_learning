#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
    y: placeholder for the labels of the input data (one-hot encoded)
    y_pred: tensor containing the network's predictions

    Returns:
    tensor containing the decimal accuracy of the prediction
    """

    correct_predictions = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
