#!/usr/bin/env python3
create_layer = __import__('1-create_layer').create_layer
"""forward propagation for a neural network"""


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for a neural network.

    Args:
    x: placeholder for the input data
    layer_sizes: number of nodes in each layer of the network
    activations: activation functions for each layer

    Returns:
    the prediction of the network
    """
    output = x

    for i in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])

    return output
