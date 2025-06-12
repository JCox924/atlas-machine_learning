#!/usr/bin/env python3
"""Module to perform random brightness adjustment on images."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly adjusts the brightness of a 3D image tensor.

    Args:
        image (tf.Tensor): 3D tensor representing the image (H, W, C).
        max_delta (float): Maximum delta for brightness adjustment. The new image
            will have values in [image - max_delta, image + max_delta].

    Returns:
        tf.Tensor: The brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(image, max_delta)
