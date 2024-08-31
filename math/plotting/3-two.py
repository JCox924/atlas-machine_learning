#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def two():
    """

    :return: Pyplot line graph with two lines
    """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    line1, = plt.plot(x, y1, 'r--', label='C-14')
    line2, = plt.plot(x, y2, 'g-', label='Ra-226')
    plt.legend(handles=[line1, line2], loc=1)
    plt.show()
