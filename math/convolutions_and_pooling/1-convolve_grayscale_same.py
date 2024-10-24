#!/usr/bin/env python3
"""
Module convolve contains the function:
    convolve_grayscale_same(images, kernel)
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images with zero padding.

    Args:
        images (numpy.ndarray): with shape (m, h, w)
            containing multiple grayscale images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
        kernel (numpy.ndarray): with shape (kh, kw) containing the
            kernel for the convolution.
            - kh is the height of the kernel.
            - kw is the width of the kernel.

    Returns:
        numpy.ndarray: containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2 if kh % 2 != 0 else (kh // 2)
    pw = kw // 2 if kw % 2 != 0 else (kw // 2)

    padded_images = np.pad(images,
                           ((0, 0),
                            (ph, ph + (kh % 2 == 0)),
                            (pw, pw + (kw % 2 == 0))),
                           mode='constant')

    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:i + kh, j:j + kw]
            convolved_images[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved_images
