#!/usr/bin/env python3
"""Train a CNN using Keras Applications for CIFAR-10 classification."""

import numpy as np
from tensorflow import keras as K
import tensorflow as tf

def preprocess_data(X, Y):
    """
    Preprocesses the data for the model.

    Parameters:
        X: np.ndarray - shape (m, 32, 32, 3), containing CIFAR-10 data
        Y: np.ndarray - shape (m,), containing CIFAR-10 labels for X

    Returns:
        X_p: np.ndarray - preprocessed X
        Y_p: np.ndarray - preprocessed Y
    """
    X_p = np.array([tf.image.resize(img, (224, 224)).numpy() for img in X])
    # Apply MobileNetV2 preprocessing
    X_p = K.applications.mobilenet_v2.preprocess_input(X_p)
    # Convert labels to one-hot encoding
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    train_dataset = preprocess_data(X_train, Y_train)
    test_dataset = preprocess_data(X_test, Y_test)

    batch_size = 32

    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    input_shape = (224, 224, 3)

    inputs = K.Input(shape=input_shape)

    base_model = K.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False

    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=10,
        verbose=1
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=5,
        verbose=1
    )

    model.save('cifar10.h5')
    print("Model saved as cifar10.h5")
