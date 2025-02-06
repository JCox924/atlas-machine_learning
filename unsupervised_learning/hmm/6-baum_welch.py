#!/usr/bin/env python3
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Parameters:
        Observations (numpy.ndarray): Array of shape (T,) containing the index
            of each observation.
        Transition (numpy.ndarray): Array of shape (M, M) containing the
            initialized transition probabilities, where M is the number of
            hidden states.
        Emission (numpy.ndarray): Array of shape (M, N) containing the
            initialized emission probabilities, where N is the number of
            output states.
        Initial (numpy.ndarray): Array of shape (M, 1) containing the
            initialized starting probabilities.
        iterations (int): Number of EM iterations to perform.

    Returns:
        Transition, Emission: The converged transition and emission matrices.
        On failure, returns (None, None).
    """
    if not (isinstance(Observations, np.ndarray) and Observations.ndim == 1 and
            isinstance(Transition, np.ndarray) and Transition.ndim == 2 and
            isinstance(Emission, np.ndarray) and Emission.ndim == 2 and
            isinstance(Initial, np.ndarray) and Initial.ndim == 2):
        return None, None

    M = Transition.shape[0]
    if (Transition.shape != (M, M) or Emission.shape[0] != M or
            Initial.shape != (M, 1)):
        return None, None

    N = Emission.shape[1]
    T = Observations.shape[0]
    if not np.all((Observations >= 0) & (Observations < N)):
        return None, None

    for _ in range(iterations):
        alpha = np.zeros((T, M))
        beta = np.zeros((T, M))
        init = Initial.flatten()
        alpha[0, :] = init * Emission[:, Observations[0]]
        for t in range(1, T):
            for j in range(M):
                alpha[t, j] = np.sum(alpha[t - 1, :] *
                                     Transition[:, j])
                alpha[t, j] *= Emission[j, Observations[t]]
        P = np.sum(alpha[T - 1, :])
        if P == 0:
            return None, None

        beta[T - 1, :] = 1
        for t in range(T - 2, -1, -1):
            for i in range(M):
                beta[t, i] = np.sum(Transition[i, :] *
                                    Emission[:, Observations[t + 1]] *
                                    beta[t + 1, :])

        gamma = (alpha * beta) / P
        xi = np.zeros((T - 1, M, M))
        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[t, i, j] = (alpha[t, i] *
                                   Transition[i, j] *
                                   Emission[j, Observations[t + 1]] *
                                   beta[t + 1, j])
            xi_sum = np.sum(xi[t, :, :])
            if xi_sum == 0:
                xi_sum = 1e-10
            xi[t, :, :] /= xi_sum

        new_T = np.zeros((M, M))
        for i in range(M):
            denom = np.sum(gamma[:-1, i])
            if denom == 0:
                denom = 1e-10
            for j in range(M):
                new_T[i, j] = np.sum(xi[:, i, j]) / denom

        new_E = np.zeros((M, N))
        for i in range(M):
            denom = np.sum(gamma[:, i])
            if denom == 0:
                denom = 1e-10
            for k in range(N):
                mask = (Observations == k)
                new_E[i, k] = np.sum(gamma[mask, i]) / denom

        Transition = new_T
        Emission = new_E
        Initial = gamma[0, :].reshape((M, 1))

    return Transition, Emission
