#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 CNN using TensorFlow
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture
    Args:
        x: tf.placeholder of shape (m, 28, 28, 1) containing input images
        y: tf.placeholder of shape (m, 10) containing one-hot labels
    Returns:
        output: tensor for softmax activated output
        train_op: training operation using Adam optimizer
        loss: tensor for the loss of the network
        accuracy: tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.keras.layers.Conv2D(
        filters=6,
        kernel_size=[5, 5],
        padding="same",
        activation='relu',
        kernel_initializer=initializer
    )(x)

    pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2
    )(conv1)

    conv2 = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2
    )(conv2)

    pool2_flat = tf.layers.Flatten()(pool2)

    fc1 = tf.keras.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(pool2_flat)

    fc2 = tf.keras.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    logits = tf.keras.layers.Dense(
        units=10,
        kernel_initializer=initializer
    )(fc2)

    output = tf.nn.softmax(logits)

    loss = (
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                                  logits=logits
                                                                  )
                       )
    )
    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, train_op, loss, accuracy
