#!/usr/bin/env python3
"""
Module 7-early_stopping contains functions:
    early_stopping(cost, opt_cost, threshold, patience, count)
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        Determines if gradient descent should be stopped early.

        Early stopping occurs when the validation
        cost has not decreased relative
        to the optimal validation cost by more than the
        threshold over a specific
        patience count.

        Args:
            cost: current validation cost of the neural network
            opt_cost: lowest recorded validation cost of the neural network
            threshold: threshold used for early stopping
            patience: patience count used for early stopping
            count: count of how long the threshold has not been met

        Returns:
            stop: boolean indicating whether to stop early
            count: updated count of how long the threshold has not been met
        """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    else:
        return False, count
