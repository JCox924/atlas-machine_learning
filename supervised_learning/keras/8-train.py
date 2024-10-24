#!/usr/bin/env python3
"""
Module train_model contains the function:
    train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional validation data, early stopping,
    learning rate decay, and saving the best iteration of the model.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data to train the model.
        labels (numpy.ndarray): One-hot encoded labels for the data.
        batch_size (int): Size of mini-batches.
        epochs (int): Number of epochs to train.
        validation_data (tuple or None): Data to validate the model on.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        learning_rate_decay (bool): Whether to use learning rate decay.
        alpha (float): Initial learning rate.
        decay_rate (float): Decay rate for learning rate decay.
        save_best (bool): Whether to save the best model iteration based on validation loss.
        filepath (str or None): Path to save the best model.
        verbose (bool): Whether to output verbose logs during training.
        shuffle (bool): Whether to shuffle the data during training.

    Returns:
        History: History object generated after training.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(early_stopping_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            """Inverse time decay learning rate schedule"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
        callbacks.append(lr_decay_callback)

    if save_best and validation_data is not None and filepath is not None:
        checkpoint_callback = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                          monitor='val_loss',
                                                          save_best_only=True,
                                                          verbose=0)
        callbacks.append(checkpoint_callback)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callbacks)

    return history
