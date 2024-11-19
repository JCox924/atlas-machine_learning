#!/usr/bin/env python3
"""Transfer learning script for CIFAR-10 classification."""

from tensorflow import keras as K
import tensorflow as tf

def preprocess_data(X, Y):
    """Preprocesses the data for the model."""
    X_p = X.astype('float32')
    X_p = tf.image.resize(X_p, (224, 224)).numpy()
    X_p = K.applications.resnet50.preprocess_input(X_p)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(224, 224, 3))

    base_model = K.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=inputs
    )
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_p, Y_train_p,
        validation_data=(X_test_p, Y_test_p),
        batch_size=128,
        epochs=10
    )

    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_p, Y_train_p,
        validation_data=(X_test_p, Y_test_p),
        batch_size=128,
        epochs=5
    )

    model.save('cifar10.h5')
