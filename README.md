# Linear Differential Equations Solver

This Python repository contains an implementation of a solver for linear differential equations using the finite
differences method.

## Overview

The solver utilizes the finite differences method to approximate solutions to linear differential equations. It is
particularly useful for solving ordinary differential equations (ODEs) of the form:

```math
a_n(x)y^{(n)}+a_{n-1}(x)y^{(n-1)}+...+a_1y'+a_0(x)y=f(x)
```

where `y` is the unknown function, `f(x)` is a given function, and
`a_i(x)`
are the known functions of `x`

# How to run

To use this solver, follow these steps:

1. Clone or download this repository to your local machine.
2. Navigate to the repository directory in your terminal.

To run the tests:

3. Ensure you have Python installed on your system.
4. Run the following command:
   `python -m unittest discover tests` \
   This command will execute the test suite located in the tests package to verify the correctness of the solver
   implementation.

## Packages

The repository is organized into the following packages:

- **solver**: Contains the implementation of the linear nonhomogeneous differential equation solver.
    - LinearNonhomogeneousDifferentialEquation: This module provides a solver with a solve method to find the solution
      to a linear nonhomogeneous differential equation.
- **tests**: Contains the test suite to validate the functionality of the solver.
  The tests are designed to verify the correctness and robustness of the solver's implementation.

## Usage

To use the solver in your own projects, you can import the `LinearNonhomogeneousDifferentialEquation` class from
the `solver` package. Here's a basic example of how to use the solver:

Assume we have an equation of type

```math
y'' + p(x)y' + q(x)y = f(x)
```

```
from solver import LinearNonhomogenousDifferentialEquation

def p(x):
    return -2 / (2 * x + 1)

def q(x):
    return -12 / (2 * x + 1) ** 2

def f(x):
    return (3 * x + 1) / (2 * x + 1) ** 2

# Instantiate a solver
solver = LinearNonhomogenousDifferentialEquation(p, q, f)

# Define a grid
grid = np.linspace(2, 6, 6)

# Define boundary conditions (values of the y function in grid borders)
boundary_conditions = (4, 8)

# Get the solution
# The solver also supports irregular grids
solution = solver.solve_with_boundary_conditions(grid, boundary_conditions, grid_type='regular', method='thomas')

# Solution is 2-dimensional ndarray with zipped x and y values
```