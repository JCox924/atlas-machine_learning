#!/usr/bin/env python3
"""
Module 5-backward contains fucntion(s):
    backward(Observation, Emission, Transition, Initial)
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Parameters:
        Observation (numpy.ndarray): Array of shape (T,) with observation
            indices.
        Emission (numpy.ndarray): Array of shape (N, M) with emission
            probabilities. Emission[i, j] is the probability of observing j
            given state i.
        Transition (numpy.ndarray): Array of shape (N, N) with transition
            probabilities. Transition[i, j] is the probability
            of transitioning
            from state i to state j.
        Initial (numpy.ndarray): Array of shape (N, 1) with initial state
            probabilities.

    Returns:
        P: The likelihood of the observation sequence given the model.
        B: A numpy.ndarray of shape (N, T) containing the backward path
           probabilities, where B[i, t] is the probability of generating the
           future observations from state i at time t.
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

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.dot(Transition,
                         Emission[:, Observation[t + 1]] *
                         B[:, t + 1])

    P = np.sum(Initial.flatten() *
               Emission[:, Observation[0]] *
               B[:, 0])
    return P, B
