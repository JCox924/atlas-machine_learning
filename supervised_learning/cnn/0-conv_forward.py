#!/usr/bin/env python3
"""
Module for performing forward propagation over a convolutional layer
"""
import numpy as np


def pad_valid(A_prev):
    """
    Performs valid padding (no padding)
    Args:
        A_prev: Input to be padded
    Returns:
        Padded input (same as input for valid padding)
    """
    return A_prev


def pad_same(A_prev, kh, kw, sh, sw):
    """
    Performs same padding on images
    Args:
        A_prev: Input to be padded
        kh: Filter height
        kw: Filter width
        sh: Stride height
        sw: Stride width
    Returns:
        Padded input
    """
    ph = max((((A_prev.shape[1] - 1) * sh + kh - A_prev.shape[1]) // 2), 0)
    pw = max((((A_prev.shape[2] - 1) * sw + kw - A_prev.shape[2]) // 2), 0)

    return np.pad(A_prev,
                  ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                  mode='constant')


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
            kh: filter height
            kw: filter width
            c_prev: number of channels in the previous layer
            c_new: number of channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution
        activation: activation function applied to the convolution
        padding: string that is either same or valid, indicating padding type
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh: stride for the height
            sw: stride for the width
    Returns:
        The output of the convolutional layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]

    sh, sw = stride

    if padding == "valid":
        padded = pad_valid(A_prev)
        ph, pw = 0, 0
    elif padding == "same":
        padded = pad_same(A_prev, kh, kw, sh, sw)
        ph = max((((h_prev - 1) * sh + kh - h_prev) // 2), 0)
        pw = max((((w_prev - 1) * sw + kw - w_prev) // 2), 0)
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, c_new))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            current_slice = padded[:, h_start:h_end, w_start:w_end, :]

            for k in range(c_new):
                conv = np.sum(
                    current_slice * W[:, :, :, k],
                    axis=(1, 2, 3)
                )
                output[:, i, j, k] = conv

    output = output + b

    if activation is not None:
        output = activation(output)

    return output