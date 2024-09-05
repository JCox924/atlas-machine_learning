#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    names = ['Farrah', 'Fred', 'Felicia']

    colors = ['#FF0000', '#FFFF00', '#FF8000', '#FFE5B4']

    labels = ['apples', 'bananas', 'oranges', 'peaches']

    width = 0.5

    indices = np.arange(len(names))

    floor = np.zeros(len(names))

    for i in range(fruit.shape[0]):
        plt.bar(indices, fruit[i], width=width, bottom=floor, color=colors[i], label=labels[i])
        floor += fruit[i]

    plt.ylabel('Quantity of Fruits')
    plt.title('Number of Fruit per Person')
    plt.xticks(indices, names)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)

    plt.legend(loc=1)
    plt.show()
