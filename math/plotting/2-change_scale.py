#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
2-change_scale
:functions: change scale
"""


def change_scale():
    """

    :return: Line PyPlot of the exponential radioactive-decay of Carbon-14
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 28651)
    plt.yscale('log')
    plt.title('Exponential Decay of C-14')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.plot(x, y, c='#0000FF')
