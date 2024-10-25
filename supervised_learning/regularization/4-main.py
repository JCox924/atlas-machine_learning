#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot


def visualize_dropout_forward_prop(cache, L):
    """
    Visualizes activations and dropout masks from forward propagation in a single plot.

    Args:
        cache (dict): Dictionary containing activations (A) and dropout masks (D)
        L (int): Number of layers in the network
    """
    # Create a figure with subplots for activations and dropout masks
    fig, axs = plt.subplots(L, 2, figsize=(12, 4 * L))

    for layer in range(1, L + 1):
        # Visualize activations
        A = cache["A" + str(layer)]
        ax = axs[layer - 1, 0] if L > 1 else axs[0]  # Handling 1-layer case
        ax.set_title(f"Layer {layer} Activations")
        img = ax.imshow(A, aspect='auto', cmap='viridis')
        fig.colorbar(img, ax=ax)
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Neurons')

        # Visualize dropout mask (only for layers before the final one)
        if layer < L:
            D = cache["D" + str(layer)]
            ax = axs[layer - 1, 1] if L > 1 else axs[1]  # Handling 1-layer case
            ax.set_title(f"Layer {layer} Dropout Mask")
            img = ax.imshow(D, aspect='auto', cmap='gray')
            fig.colorbar(img, ax=ax)
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Neurons')
        else:
            # Last layer doesn't have dropout mask
            axs[layer - 1, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    lib = np.load('../optimization/MNIST.npz')
    X_train_3D = lib['x_train']
    Y_train = lib['y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    # Perform dropout forward propagation
    cache = dropout_forward_prop(X_train, weights, 3, 0.8)

    # Visualize the activations and dropout masks
    visualize_dropout_forward_prop(cache, 3)

    # Optionally print the cached activations and dropout masks
    for k, v in sorted(cache.items()):
        print(k, v)
