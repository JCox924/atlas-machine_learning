#!/usr/bin/env python3
"""
Module pool contains the function:
    pool(images, kernel_shape, stride, mode='max')
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (numpy.ndarray): with shape (m, h, w, c)
            containing multiple images.
            - m is the number of images.
            - h is the height in pixels of the images.
            - w is the width in pixels of the images.
            - c is the number of channels in the image.
        kernel_shape (tuple): (kh, kw) containing
            the kernel shape for the pooling.
            - kh is the height of the kernel.
            - kw is the width of the kernel.
        stride (tuple): (sh, sw) where:
            - sh is the stride for the height of the image.
            - sw is the stride for the width of the image.
        mode (str): 'max' for max pooling or 'avg' for average pooling.

    Returns:
        numpy.ndarray: containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    pooled_images = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            vert_st = i * sh
            vert_e = vert_st + kh
            horiz_st = j * sw
            horiz_e = horiz_st + kw

            if mode == 'max':
                pooled_images[:, i, j, :] = (
                    np.max(
                        images[:, vert_st:vert_e, horiz_st:horiz_e, :],
                        axis=(1, 2)))
            elif mode == 'avg':
                pooled_images[:, i, j, :] =\
                    np.mean(
                        images[:, vert_st:vert_e, horiz_st:horiz_e, :],
                        axis=(1, 2))

    return pooled_images
