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
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    fc1 = tf.layers.dense(
        inputs=pool2_flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    fc2 = tf.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    output = tf.layers.dense(
        inputs=fc2,
        units=10,
        activation=tf.nn.softmax,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    loss = (
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                                  logits=output
                                                                  )
                       )
    )
    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, train_op, loss, accuracy
