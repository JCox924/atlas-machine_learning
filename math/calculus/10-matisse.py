#!/usr/bin/env python3

def poly_derivative(poly):
    if not isinstance(poly, list) or len(poly) == 0 \
            or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]

    derivative = [coef * i for i, coef in enumerate(poly) if i > 0]

    return derivative if derivative else [0]
