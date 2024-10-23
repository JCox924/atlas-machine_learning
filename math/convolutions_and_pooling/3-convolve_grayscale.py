#!/usr/bin/env python3
"""
Module convolve contains the function:
    convolve_grayscale(images, kernel, padding='same', stride=(1, 1))
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride

    Args:
        images (numpy.ndarray): with shape (m, h, w) containing
            multiple grayscale images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
        kernel (numpy.ndarray): with shape (kh, kw) containing
            the kernel for the convolution.
            - kh is the height of the kernel.
            - kw is the width of the kernel.
        padding (str or tuple): either a tuple of (ph, pw), 'same', or 'valid'.
            - 'same': performs a same convolution.
            - 'valid': performs a valid convolution.
            - tuple: (ph, pw) is the padding for the
                height and width of the image.
        stride (tuple): (sh, sw) where:
            - sh is the stride for the height of the image.
            - sw is the stride for the width of the image.

    Returns:
        numpy.ndarray: containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           ((0, 0),
                            (ph, ph),
                            (pw, pw)),
                           mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    convolved_images = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            convolved_images[:, i, j] =\
                np.sum(padded_images[:, vert_start:vert_end, horiz_start:horiz_end]
                       * kernel,
                       axis=(1, 2))

    return convolved_images
