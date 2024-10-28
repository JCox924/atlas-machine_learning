#!/usr/bin/env python3
"""
Module for performing back propagation over a pooling layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer
    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c)
            containing the partial derivatives with respect to output
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c: number of channels
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
                containing the output of the previous layer
            h_prev: height of the previous layer
            w_prev: width of the previous layer
        kernel_shape: tuple of (kh, kw) containing kernel size
            kh: kernel height
            kw: kernel width
        stride: tuple of (sh, sw) containing the strides
            sh: stride for the height
            sw: stride for the width
        mode: string containing either max or avg
            indicates the type of pooling
    Returns:
        The partial derivatives with respect to the previous layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c = A_prev.shape[3]

    h_new = dA.shape[1]
    w_new = dA.shape[2]

    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    a_prev_slice = A_prev[
                                   i, h_start:h_end, w_start:w_end, ch
                                   ]

                    if mode == 'max':
                        mask = (a_prev_slice == np.max(a_prev_slice))

                        dA_prev[i, h_start:h_end, w_start:w_end, ch] += (
                                mask * dA[i, h, w, ch]
                        )

                    elif mode == 'avg':
                        average = dA[i, h, w, ch] / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end, ch] += (
                                np.ones(kernel_shape) * average
                        )

                    else:
                        raise ValueError("mode must be 'max' or 'avg'")

    return dA_prev
