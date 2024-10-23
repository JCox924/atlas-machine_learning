#!/usr/bin/env python3
"""
Module optimize_model contains function:
    optimize_model(network, alpha, beta1, beta2)
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical
        cross-entropy loss and accuracy metrics.

    Args:
        network: The model to optimize.
        alpha: The learning rate.
        beta1: The first Adam optimization parameter (momentum term).
        beta2: The second Adam optimization parameter (momentum term).

    Returns: None
    """
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                         beta_1=beta1,
                                         beta_2=beta2)

    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
