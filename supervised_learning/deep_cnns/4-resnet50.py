#!/usr/bin/env python3
"""
Module 4-resnet50 contains function:
    resnet50()
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
        'Deep Residual Learning for Image Recognition' (2015).

    Returns:
        model: the keras Model
    """
    initializer = K.initializers.HeNormal(seed=0)

    input_layer = K.Input(shape=(224, 224, 3))

    # Initial Convolution and max pooling
    x = K.layers.Conv2D(64,
                        kernel_size=7,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer)(input_layer)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.ReLU()(x)
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Convolution Stage 2
    x = projection_block(x, (64, 64, 256), s=1)
    x = identity_block(x, (64, 64, 256))
    x = identity_block(x, (64, 64, 256))

    # Convolution Stage 3
    x = projection_block(x, (128, 128, 512), s=2)
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))

    # Convolution Stage 4
    x = projection_block(x, (256, 256, 1024), s=2)
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))

    # Convolution Stage 5
    x = projection_block(x, (512, 512, 2048), s=2)
    x = identity_block(x, (512, 512, 2048))
    x = identity_block(x, (512, 512, 2048))

    x = K.layers.AveragePooling2D(pool_size=7, strides=1)(x)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=initializer)(x)

    model = K.Model(inputs=input_layer, outputs=output)

    return model
