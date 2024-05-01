import numpy as np
from matplotlib import pyplot


class LinearNonhomogenousDifferentialEquation:
    def __init__(self, p, q, f):
        self._p = p
        self._q = q
        self._f = f

    def solve_with_initial_condintions(self, start, stop, accuracy: int, method='runge_kutta'):
        raise NotImplementedError()

    def solve_with_boundary_conditions(self, grid, boundary_conditions, grid_type='regular',
                                       method='thomas'):
        if grid_type == 'regular':
            n = len(grid)
            h = (grid[n - 1] - grid[0]) / n

            A = np.zeros(n)
            C = np.zeros(n)
            B = np.zeros(n)
            F = np.zeros(n)

            A[0] = 0
            C[0] = -1
            B[0] = 0
            F[0] = self._F_i(grid[0], h)
            for i in range(1, n - 1):
                A[i] = self._A_i(grid[i], h)
                C[i] = self._C_i(grid[i], h)
                B[i] = self._B_i(grid[i], h)
                F[i] = self._F_i(grid[i], h)
            C[n - 1] = -1
            A[n - 1] = 0
            B[n - 1] = 0
            F[n - 1] = self._F_i(grid[n - 1], h)

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
        alpha = np.zeros(n)
        beta = np.zeros(n)
        alpha[0] = - B[0] / C[0]
        beta[0] = - F[0] / C[0]
        for i in range(1, n):
            alpha[i] = B[i] / (C[i] - A[i] * alpha[i - 1])
            beta[i] = (A[i] * beta[i - 1] - F[i]) / (C[i] - A[i] * alpha[i - 1])

        Y = np.zeros(n)
        Y[0] = boundary_conditions[0]
        Y[n - 1] = boundary_conditions[1]
        for i in reversed(range(1, n)):
            Y[i - 1] = alpha[i - 1] * Y[i] + beta[i - 1]

        return Y

    @staticmethod
    def compare_solutions(model_solution, test_solution):
        """Compares two solutions using mean squared error"""
        # interp_func = interpolate.interp1d(test_solution[:, 0], test_solution[:, 1], kind='linear')
        # interpolated_solution2 = interp_func(model_solution[:, 0])

        interpolated_solution = np.interp(model_solution[:, 0], test_solution[:, 0], test_solution[:, 1])

        # pyplot.plot(model_solution[:, 0], interpolated_solution, color='brown')
        difference = np.abs(model_solution[:, 1] - interpolated_solution)
        # pyplot.plot(model_solution[:, 0], difference, color='red')
        mean_difference = np.mean(difference)

        return mean_difference

    def _A_i(self, x_i, h):
        return 1 - self._p(x_i) * h / 2

    def _B_i(self, x_i, h):
        return 1 + self._p(x_i) * h / 2

    def _C_i(self, x_i, h):
        return 2 - self._q(x_i) * (h ** 2)

    def _F_i(self, x_i, h):
        return self._f(x_i) * (h ** 2)


def y(x):
    return 2 * np.exp(3 * x) + 3 * np.exp(4 * x) + 3 * x * np.exp(4 * x)


def p(x):
    return -7
    # return -2 / (2 * x + 1)


def q(x):
    return 12
    # return -12 / (2 * x + 1) ** 2


def f(x):
    return 3 * np.exp(4 * x)
    # return (3 * x + 1) / (2 * x + 1) ** 2


test_grid = np.linspace(1, 4, 100)
test_boundary_conditions = (2 * np.exp(3) + 6 * np.exp(4), 2 * np.exp(12) + 15 * np.exp(16))

solver = LinearNonhomogenousDifferentialEquation(p, q, f)
test_solution = solver.solve_with_boundary_conditions(test_grid, test_boundary_conditions)
model_solution = np.zeros((10000, 2))
for i, x in enumerate(np.linspace(1, 4, 10000)):
    model_solution[i] = [x, y(x)]
error = np.zeros(300)
for i in range(1, 300):
    grid = np.linspace(1, 4, i)
    test_solution = solver.solve_with_boundary_conditions(grid, test_boundary_conditions)
    error[i - 1] = solver.compare_solutions(model_solution, test_solution)
# print(solver.compare_solutions(model_solution, test_solution))
pyplot.plot(np.linspace(1, 300, 300), error, color='brown')
pyplot.show()

grid = np.linspace(1, 4, 1_000_000)
test_solution = solver.solve_with_boundary_conditions(grid, test_boundary_conditions)

pyplot.plot(test_solution[:, 0], test_solution[:, 1])
pyplot.plot(model_solution[:, 0], model_solution[:, 1])
pyplot.show()
