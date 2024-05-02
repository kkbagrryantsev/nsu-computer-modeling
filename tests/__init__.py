import unittest
from timeit import timeit

import numpy as np
from matplotlib import pyplot

from sample import LinearNonhomogenousDifferentialEquation


class DemoSolverTest(unittest.TestCase):
    def setUp(self):
        def p(x):
            return -2 / (2 * x + 1)

        def q(x):
            return -12 / (2 * x + 1) ** 2

        def f(x):
            return (3 * x + 1) / (2 * x + 1) ** 2

        self.solver = LinearNonhomogenousDifferentialEquation(p, q, f)

    def test_solve_demo(self):
        grid = np.linspace(2, 6, 6)
        boundary_conditions = (4, 8)

        solution = self.solver.solve_with_boundary_conditions(grid, boundary_conditions)

        pyplot.plot(solution[:, 0], solution[:, 1])
        pyplot.show()


class ModelSolverTest(unittest.TestCase):
    def setUp(self):
        def p(x):
            return -7

        def q(x):
            return 12

        def f(x):
            return 3 * np.exp(4 * x)

        def y(x):
            return 2 * np.exp(3 * x) + 3 * np.exp(4 * x) + 3 * x * np.exp(4 * x)

        self.boundary_conditions = (2 * np.exp(3) + 6 * np.exp(4), 2 * np.exp(12) + 15 * np.exp(16))

        self.solver = LinearNonhomogenousDifferentialEquation(p, q, f)

        # Model function etc
        self.model_function = y
        # Amount of calculated points
        self.model_accuracy = 100_000
        self.model_grid = np.linspace(1, 4, self.model_accuracy)
        model_y = np.vectorize(self.model_function)(self.model_grid)

        self.model_solution = np.column_stack((self.model_grid, model_y))

    def test_solve_simple_grid(self):
        grid = np.linspace(1, 4, 100)

        solution = self.solver.solve_with_boundary_conditions(grid, self.boundary_conditions)

        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        # Plot the approximated function
        ax.plot(solution[:, 0], solution[:, 1])
        # Plot the model function
        ax.plot(self.model_solution[:, 0], self.model_solution[:, 1])
        ax.set_title("Approximated & model functions comparison")
        fig.show()

    def test_relative_error(self):
        accuracy_measures = range(2, 200, 1)

        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)

        for acc in accuracy_measures:
            grid = np.linspace(1, 4, acc)

            solution = self.solver.solve_with_boundary_conditions(grid, self.boundary_conditions)

            relative_error = self.solver.compare_solutions(self.model_solution, solution, method='relative_error')
            ax.plot(self.model_grid, relative_error, color='darkred')

        # grid = np.linspace(1, 4, 100)

        # solution = self.solver.solve_with_boundary_conditions(grid, self.boundary_conditions)

        # relative_error = self.solver.compare_solutions(self.model_solution, solution, method='relative_error')

        # fig = pyplot.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(self.model_grid, relative_error, color='darkred')
        ax.set_title("Relative error")
        fig.show()

    def test_mae(self):
        # Grid of accuracy values
        accuracy_measures = range(2, 200, 10)
        mae_errors = np.zeros((len(accuracy_measures), 2))
        mse_errors = np.zeros((len(accuracy_measures), 2))
        rmse_errors = np.zeros((len(accuracy_measures), 2))
        max_absolute_errors = np.zeros((len(accuracy_measures), 2))

        # Approximation of model function on several grids
        for inx, acc in enumerate(accuracy_measures):
            grid = np.linspace(1, 4, acc)
            solution = self.solver.solve_with_boundary_conditions(grid, self.boundary_conditions)

            mae_error = self.solver.compare_solutions(self.model_solution, solution, method='mae')
            mae_errors[inx][0] = acc
            mae_errors[inx][1] = mae_error

            mse_error = self.solver.compare_solutions(self.model_solution, solution, method='mse')
            mse_errors[inx][0] = acc
            mse_errors[inx][1] = mse_error

            rmse_error = self.solver.compare_solutions(self.model_solution, solution, method='rmse')
            rmse_errors[inx][0] = acc
            rmse_errors[inx][1] = rmse_error

            max_absolute_error = self.solver.compare_solutions(self.model_solution, solution,
                                                               method='max_absolute_error')
            max_absolute_errors[inx][0] = acc
            max_absolute_errors[inx][1] = max_absolute_error

        fig = pyplot.figure(figsize=(8, 6))

        # Plot MAE
        mae_plot = fig.add_subplot(2, 2, 1)
        mae_plot.plot(mae_errors[:, 0], mae_errors[:, 1], color='darkred')
        mae_plot.set_title("MAE")

        # Plot MSE
        mse_plot = fig.add_subplot(2, 2, 2)
        mse_plot.plot(mse_errors[:, 0], mse_errors[:, 1], color='darkred')
        mse_plot.set_title("MSE")

        # Plot RMSE
        rmse_plot = fig.add_subplot(2, 2, 3)
        rmse_plot.plot(rmse_errors[:, 0], rmse_errors[:, 1], color='darkred')
        rmse_plot.set_title("RMSE")

        # Plot max absolute error
        max_plot = fig.add_subplot(2, 2, 4)
        max_plot.plot(max_absolute_errors[:, 0], max_absolute_errors[:, 1], color='darkred')
        max_plot.set_title("Max absolute error")

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.show()

    def solve(self, accuracy):
        grid = np.linspace(1, 4, accuracy)
        solution = self.solver.solve_with_boundary_conditions(grid, self.boundary_conditions)

        return solution

    def test_time(self):
        execution_time = timeit(lambda: self.solve(1000), globals=globals(), number=3)
        print("Execution time: ", execution_time, "sec for 1000")

        execution_time = timeit(lambda: self.solve(10_000), globals=globals(), number=3)
        print("Execution time: ", execution_time, "sec for 10 000")

        execution_time = timeit(lambda: self.solve(100_000), globals=globals(), number=3)
        print("Execution time: ", execution_time, "sec for 100 000")

        execution_time = timeit(lambda: self.solve(1_000_000), globals=globals(), number=3)
        print("Execution time: ", execution_time, "sec for 1 000 000")

        execution_time = timeit(lambda: self.solve(10_000_000), globals=globals(), number=3)
        print("Execution time: ", execution_time, "sec for 10 000 000")
