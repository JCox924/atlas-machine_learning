#!/usr/bin/env python3
"""Module to perform random contrast adjustment on images."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of a 3D image tensor.

    Args:
        image (tf.Tensor): 3D tensor representing the image (H, W, C).
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.

    Returns:
        tf.Tensor: The contrast-adjusted image tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
