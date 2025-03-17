# RNN

This project contains an implementation of a simple RNN cell in Python.

## Overview

The `RNNCell` class represents a cell of a simple RNN. It initializes weights and biases for both the hidden state (using a random normal distribution and zeros respectively) and the output. The cell computes the next hidden state using a tanh activation function and produces the output using a softmax activation function.

## Requirements

- **Python Version:** 3.9
- **NumPy Version:** 1.25.2
- **Operating System:** Ubuntu 20.04 LTS

## Files

- **rnn_cell.py:** Contains the implementation of the `RNNCell` class.
- **README.md:** This file.

## Usage

Make sure the file `rnn_cell.py` is executable. You can run it using:

```bash
./rnn_cell.py
