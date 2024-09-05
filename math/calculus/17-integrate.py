#!/usr/bin/env python3

def poly_integral(poly, C=0):
    # Check if poly is a valid list of numbers and C is a valid integer
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
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
