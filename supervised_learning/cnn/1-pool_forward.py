#!/usr/bin/env python3
"""
Module for performing forward propagation over a pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the kernel size
            kh: kernel height
            kw: kernel width
        stride: tuple of (sh, sw) containing the strides
            sh: stride for the height
            sw: stride for the width
        mode: string containing either max or avg
            indicates the type of pooling
    Returns:
        The output of the pooling layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            current_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                # Perform max pooling
                output[:, i, j, :] = np.max(current_slice, axis=(1, 2))
            elif mode == 'avg':
                # Perform average pooling
                output[:, i, j, :] = np.mean(current_slice, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output
