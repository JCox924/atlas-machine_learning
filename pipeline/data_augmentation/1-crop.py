#!/usr/bin/env python3
"""Module to perform random cropping on images."""

import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of a 3D image tensor.

    Args:
        image (tf.Tensor): 3D tensor representing the image (H, W, C).
        size (tuple): Tuple (crop_height, crop_width) for the output size.

    Returns:
        tf.Tensor: The randomly cropped image tensor of shape
            (crop_height, crop_width, C).
    """
    crop_height, crop_width = size
    channels = tf.shape(image)[-1]
    return tf.image.random_crop(image,
                                size=(crop_height,
                                      crop_width,
                                      channels))
