#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
""" This module has a function that shows a scatter plot """


def scatter():
    """

    :return: Shows a Scatter plot of Men's Height vs Weight
    """

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, color='#FF00FF')
    plt.title("Men's Height vs Weight")
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.show()