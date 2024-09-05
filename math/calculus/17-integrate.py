#!/usr/bin/env python3
"""
This module contains the function(s) poly_integral(poly, C=0)
"""


def poly_integral(poly, C=0):
    """

    :arg poly: list of polynomials
    :arg C: constant
    :return: integral
    """
    if not isinstance(poly, list) or len(poly) == 0 or \
            not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, int):
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        integral_coef = coef / (i + 1)
        if integral_coef.is_integer():
            integral_coef = int(integral_coef)
        integral.append(integral_coef)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
