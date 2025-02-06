#!/usr/bin/env python3
"""
Module 4-viterbi contains
    function(s):
        viterbi(Observation, Emission, Transition, Initial)
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    Markov model using the Viterbi algorithm.

    Parameters:
        Observation (numpy.ndarray): Array of shape (T,) containing the
            observation indices.
        Emission (numpy.ndarray): Array of shape (N, M) with emission
            probabilities. Emission[i, j] is the probability of observing j
            given hidden state i.
        Transition (numpy.ndarray): Array of shape (N, N) with transition
            probabilities. Transition[i, j] is the probability of transitioning
            from state i to state j.
        Initial (numpy.ndarray): Array of shape (N, 1) with initial state
            probabilities.

    Returns:
        path: A list of length T containing the most likely sequence of hidden
            states.
        P: The probability of obtaining the path sequence.
        On failure, returns (None, None).
    """
    if not (isinstance(Observation, np.ndarray) and
            Observation.ndim == 1 and
            isinstance(Emission, np.ndarray) and Emission.ndim == 2 and
            isinstance(Transition, np.ndarray) and Transition.ndim == 2 and
            isinstance(Initial, np.ndarray) and Initial.ndim == 2):
        return None, None

    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    T = Observation.shape[0]
    if not np.all((Observation >= 0) & (Observation < M)):
        return None, None

    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=int)
    init = Initial.flatten()
    delta[:, 0] = init * Emission[:, Observation[0]]
    psi[:, 0] = 0

    for t in range(1, T):
        for i in range(N):
            prod = delta[:, t - 1] * Transition[:, i]
            psi[i, t] = np.argmax(prod)
            delta[i, t] = prod[psi[i, t]] * Emission[i, Observation[t]]

    last_state = int(np.argmax(delta[:, T - 1]))
    P = float(np.max(delta[:, T - 1]))

    path = [0] * T
    path[T - 1] = last_state
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]

    return path, P
