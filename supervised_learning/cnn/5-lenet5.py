#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 CNN using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras
    Args:
        X: K.Input of shape (m, 28, 28, 1) containing input images
    Returns:
        model: K.Model compiled with Adam optimizer and accuracy metrics
    """
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=[5, 5],
        padding="same",
        activation=K.activations.relu,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0,
                                                          mode='fan_in',
                                                          distribution='normal',
                                                          seed=0)
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=[5, 5],
        padding="valid",
        activation=K.activations.relu,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0,
                                                          mode='fan_in',
                                                          distribution='normal',
                                                          seed=0)
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2
    )(conv2)

    pool2_flat = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        units=120,
        activation=K.activations.relu,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0,
                                                          mode='fan_in',
                                                          distribution='normal',
                                                          seed=0)
    )(pool2_flat)

    fc2 = K.layers.Dense(
        units=84,
        activation=K.activations.relu,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0,
                                                          mode='fan_in',
                                                          distribution='normal',
                                                          seed=0)
    )(fc1)

    output = K.layers.Dense(
        units=10,
        activation=K.activations.softmax,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0,
                                                          mode='fan_in',
                                                          distribution='normal',
                                                          seed=0)
    )(fc2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss=K.losses.categorical_crossentropy,
        metrics=[K.metrics.accuracy]
    )

    return model
