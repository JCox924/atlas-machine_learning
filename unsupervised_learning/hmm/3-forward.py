#!/usr/bin/env python3
"""
Module 3-forward contains
    function(s):
        forward(Observation, Emission, Transition, Initial)
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Arguments:
        Observation (numpy.ndarray): A numpy.ndarray of shape (T,) that contains
            the index of each observation.
        Emission (numpy.ndarray): A numpy.ndarray of shape (N, M) containing the
            emission probabilities. Emission[i, j] is the probability of observing j
            given the hidden state i.
        Transition (numpy.ndarray): A 2D numpy.ndarray of shape (N, N) containing the
            transition probabilities. Transition[i, j] is the probability of transitioning
            from hidden state i to j.
        Initial (numpy.ndarray): A numpy.ndarray of shape (N, 1) containing the initial
            state probabilities.

    Returns:
        P: The likelihood of the observations given the model.
        F: A numpy.ndarray of shape (N, T) containing the forward path probabilities,
           where F[i, t] is the probability of being in hidden state i at time t given
           the previous observations.
        If any error occurs or inputs are invalid, returns (None, None).
    """
    if not (isinstance(Observation, np.ndarray) and Observation.ndim == 1 and
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

    F = np.zeros((N, T))

    init = Initial.flatten()
    F[:, 0] = init * Emission[:, Observation[0]]

    for t in range(1, T):
        intermediate = np.dot(F[:, t-1], Transition)
        F[:, t] = intermediate * Emission[:, Observation[t]]

    P = np.sum(F[:, T-1])

    return P, F
