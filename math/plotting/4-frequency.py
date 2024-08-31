#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='#FFFFFF')
    plt.title("Project A")
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    max_y = plt.gca().get_ylim()[1]
    plt.yticks(np.arange(0, max_y + 1, 5))
    plt.xticks(np.arange(0, 101, 10))

    plt.show()
