__all__ = ['LinearNonhomogenousDifferentialEquation']

import math

import numpy as np


class LinearNonhomogenousDifferentialEquation:
    def __init__(self, p, q, f):
        self._p = p
        self._q = q
        self._f = f

    def solve_with_initial_condintions(self, start, stop, accuracy: int, method='runge_kutta'):
        raise NotImplementedError()

    def _calculate_coeffs(self, grid, boundary_conditions, grid_type='regular'):
        n = len(grid)

        A = np.zeros(n)
        C = np.zeros(n)
        B = np.zeros(n)
        F = np.zeros(n)

        A[0] = 0
        C[0] = -1
        B[0] = 0
        F[0] = boundary_conditions[0]

        A[n - 1] = 0
        C[n - 1] = -1
        B[n - 1] = 0
        F[n - 1] = boundary_conditions[-1]

        match grid_type:
            case 'regular':
                h = (grid[-1] - grid[0]) / (n - 1)
                for i in range(1, n - 1):
                    A[i] = self._regular_A_i(grid[i], h)
                    C[i] = self._regular_C_i(grid[i], h)
                    B[i] = self._regular_B_i(grid[i], h)
                    F[i] = self._regular_F_i(grid[i], h)
                return A, C, B, F

            case 'irregular':
                for i in range(1, n - 1):
                    delta_x_i = grid[i] - grid[i - 1]
                    delta_x_j = grid[i + 1] - grid[i]
                    A[i] = self._irregular_A_i(grid[i], delta_x_i, delta_x_j)
                    C[i] = self._irregular_C_i(grid[i], delta_x_i, delta_x_j)
                    B[i] = self._irregular_B_i(grid[i], delta_x_i, delta_x_j)
                    F[i] = self._irregular_F_i(grid[i], delta_x_i, delta_x_j)
                return A, C, B, F
            case _:
                raise ValueError(f'Grid type {grid_type} not supported')

    def solve_with_boundary_conditions(self, grid, boundary_conditions, grid_type='regular',
                                       method='thomas'):
        n = len(grid)

        A, C, B, F = self._calculate_coeffs(grid, boundary_conditions, grid_type)

        match method:
            case 'thomas':
                Y = LinearNonhomogenousDifferentialEquation.thomas_algorithm(A, C, B, F, boundary_conditions)
                result = np.zeros((n, 2))
                for i, x in enumerate(grid):
                    result[i] = [x, Y[i]]
                return result
            case _:
                raise ValueError(f'Method {method} not supported')

    @staticmethod
    def thomas_algorithm(A, C, B, F, boundary_conditions):
        assert A.shape == C.shape == B.shape == F.shape

        n = len(A)

        # Forward sweep
        alpha = np.zeros(n)
        beta = np.zeros(n)
        alpha[0] = - B[0] / C[0]
        beta[0] = - F[0] / C[0]
        for i in range(1, n):
            alpha[i] = B[i] / (C[i] - A[i] * alpha[i - 1])
            beta[i] = (A[i] * beta[i - 1] - F[i]) / (C[i] - A[i] * alpha[i - 1])

        # Backward sweep
        Y = np.zeros(n)
        Y[0] = boundary_conditions[0]
        Y[n - 1] = boundary_conditions[1]
        for i in reversed(range(1, n)):
            Y[i - 1] = alpha[i - 1] * Y[i] + beta[i - 1]

        return Y

    @staticmethod
    def _linear_function(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        return lambda x: slope * x + intercept

    @staticmethod
    def compare_solutions(model_solution, test_solution, method='mae'):
        """Compares two solutions using one of the specified methods"""
        assert model_solution[:, 0].min() == test_solution[:, 0].min()
        assert model_solution[:, 0].max() == test_solution[:, 0].max()

        if model_solution[:, 0].size >= test_solution[:, 0].size:
            interpolated_solution = test_solution
            solution = model_solution
        else:
            interpolated_solution = model_solution
            solution = test_solution

        if model_solution.shape[0] == test_solution.shape[0]:
            interpolated_values = interpolated_solution[:, 1]
        else:
            interpolated_values = np.interp(solution[:, 0], interpolated_solution[:, 0], interpolated_solution[:, 1])

        match method:
            case 'mae':
                difference = np.abs(solution[:, 1] - interpolated_values)
                return np.mean(difference)
            case 'mse':
                squared_difference = np.square(solution[:, 1] - interpolated_values)
                return np.mean(squared_difference)
            case 'rmse':
                squared_difference = np.square(solution[:, 1] - interpolated_values)
                mean_squared_difference = np.mean(squared_difference)
                return math.sqrt(mean_squared_difference)
            case 'relative_error':
                return np.abs(solution[:, 1] - interpolated_values) / np.abs(solution[:, 1]) * 100
            case 'max_absolute_error':
                return np.max(np.abs(solution[:, 1] - interpolated_values))
            case _:
                raise ValueError(f'Method {method} not supported')

    def _irregular_A_i(self, x_i, delta_x_i, delta_x_j):
        return 1 - self._p(x_i) * delta_x_i * delta_x_j / (delta_x_i + delta_x_j)

    def _regular_A_i(self, x_i, h):
        return 1 - self._p(x_i) * h / 2

    def _irregular_B_i(self, x_i, delta_x_i, delta_x_j):
        return 1 + self._p(x_i) * delta_x_i * delta_x_j / (delta_x_i + delta_x_j)

    def _regular_B_i(self, x_i, h):
        return 1 + self._p(x_i) * h / 2

    def _irregular_C_i(self, x_i, delta_x_i, delta_x_j):
        return 2 - self._q(x_i) * delta_x_i * delta_x_j

    def _regular_C_i(self, x_i, h):
        return 2 - self._q(x_i) * (h ** 2)

    def _irregular_F_i(self, x_i, delta_x_i, delta_x_j):
        return self._f(x_i) * delta_x_i * delta_x_j

    def _regular_F_i(self, x_i, h):
        return self._f(x_i) * (h ** 2)
