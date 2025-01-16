# Overview
This repository contains implementations of mathematical operations and concepts related to matrices. Each file represents a specific operation or property, providing Python functions that adhere to clean coding practices and mathematical rigor.

## Files and Descriptions

0. [Determinant]()

**Purpose**: Computes the determinant of a square matrix.

Functionality:

    Input: A square matrix represented as a list of lists.
    Output: The determinant of the matrix as a scalar value.

Example Use Case: Determining the invertibility of a matrix.

1. [Minor]()

**Purpose**: Calculates the minor of a specific element in a matrix.

Functionality:

    Input: A matrix and the row and column indices of the element.
    Output: The determinant of the submatrix formed by removing the specified row and column.

Example Use Case: Foundational step in computing the cofactor matrix.
2. [Cofactor]()

**Purpose**: Computes the cofactor matrix of a given square matrix.

Functionality:

    Input: A square matrix.
    Output: The cofactor matrix.

Example Use Case: Used to calculate the adjugate and inverse of a matrix.

3. [Adjugate]()

**Purpose**: Computes the adjugate (adjoint) of a square matrix.

Functionality:

    Input: A square matrix.
    Output: The adjugate matrix, which is the transpose of the cofactor matrix.

Example Use Case: Integral to finding the inverse of a matrix.

4. [Inverse]()

**Purpose**: Calculates the inverse of a square matrix, if it exists.

Functionality:

    Input: A square matrix.
    Output: The inverse of the matrix, or an error if the matrix is singular (non-invertible).

Example Use Case: Solving systems of linear equations.

5. [Definiteness]()

**Purpose**: Determines the definiteness (positive definite, negative definite, etc.) of a matrix.

Functionality:

    Input: A square, symmetric matrix.
    Output: A string indicating the definiteness of the matrix.

Example Use Case: Assessing stability in optimization and systems of equations.

## Requirements

### Python Version: Tested on Python 3.8+

### Dependencies:

None (standard Python library is sufficient).

### License

This project is licensed under the MIT License. See the LICENSE file for details.
