#!/usr/bin/env python3

import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

Neuron = __import__('5-neuron').Neuron

file_path = 'Binary_Train.npz'

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file '{file_path}' does not exist. Please ensure the file is in the correct directory.")

lib_train = np.load(file_path)
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)