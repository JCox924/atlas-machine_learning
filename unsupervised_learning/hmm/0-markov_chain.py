#!/usr/bin/env python3
"""
Module 0-markov_chain contains
    function:
        markov_chain(P, s t=1):
"""
import numpy as np


def markov_chain(P, s, t=1) -> np.ndarray:
    """
    Determine the markov chain of P with probability s

    Arguments:
        P (numpy.ndarray):
            square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix
                P[i, j], is the probability of transitioning
                    from state i to state j
                n, is the number of states in the markov chain
        s (numpy.ndarray):
            ndarray of shape (1, n) representing
            the probability of starting in each state
        t (int): The number of iterations (transitions) of the Markov chain.

    Returns:
         (numpy.ndarray): (1, n) shape matrix  representing the probability
            of being in a specific state after t iterations,
            or None on failure
    """
    if (not isinstance(P, np.ndarray)) or (not isinstance(s, np.ndarray) or
            P.dim != 2 or s.ndim != 2 or P.shape[0] != P.shape[1] or
            s.shape[0] != 1 or s.shape[1] != P.shape[0] or
            not isinstance(t, int) or t < 0):
        return None

    try:
        P_t = np.linalg.matrix_power(P, t)
        m_chain = np.dot(s, P_t)
    except Exception:
        return None

    return m_chain
