#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

"""This file module contains the function(s) line() with shows a Line Plot"""


def line():
    """
    line(void)
    :return: Line PyPlot
    """
    x = np.arange(0, 11)
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='#FF0000')
    plt.xlim(0, 10)
    plt.show()
