#!/usr/bin/env python3
"""
Module train_model contains function:
    train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional validation data, early stopping, and learning rate decay.

    Args:
        network: The model to train.
        data: A numpy.ndarray of shape (m, nx) containing the input data.
        labels: A one-hot numpy.ndarray of shape (m, classes) containing the labels of data.
        batch_size: The size of the batch used for mini-batch gradient descent.
        epochs: The number of passes through data for mini-batch gradient descent.
        validation_data: The data to validate the model with, if not None.
        early_stopping: A boolean that indicates whether early stopping should be used.
        patience: The patience for early stopping (number of epochs with no improvement to wait).
        learning_rate_decay: A boolean that indicates whether learning rate decay should be used.
        alpha: The initial learning rate.
        decay_rate: The decay rate for inverse time decay.
        verbose: A boolean that determines if output should be printed during training.
        shuffle: A boolean that determines whether to shuffle the batches every epoch.

    Returns:
        History: The History object generated after training the model.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(early_stopping_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            """Inverse time decay learning rate schedule"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_decay_callback)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data, verbose=verbose,
                          shuffle=shuffle, callbacks=callbacks)

    return history
