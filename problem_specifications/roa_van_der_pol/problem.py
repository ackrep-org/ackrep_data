import numpy as np
from ackrep_core import ResultContainer
import scipy.integrate as sc_integrate
from matplotlib import pyplot as plt

import symbtools.meshtools as met


class ProblemSpecification(object):
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)
    x_init_res = 5
    y_init_res = 5
    max_refinement_steps = 5
    tt = np.linspace(0.0, 50.0, 5000)

    @staticmethod
    def sys_rhs(t, x):
        """
        negative Zeit-Richtung Funktion, später wird es in "judge-funktion" benutzt.
        rhs funktion. gegengesetzten Zeitrichtung.
        """
        mu = 2

        dx0 = -1 * x[1]
        dx1 = -1 * (-mu * (x[0] ** 2.0 - 1.0) * x[1] - x[0])
        res = np.array([dx0, dx1])
        return res

    @staticmethod
    def has_converged(t, y):
        return np.sum(y ** 2) <= 0.1

    @staticmethod
    def has_diverged(t, y):
        return np.sum(y ** 2) > 100


def evaluate_solution(solution_data):
    grid_volumes = np.array(solution_data.grid.sum_volumes())
    target_volumes = np.array([15.0, 81.4, 3.6])
    success = np.sum((grid_volumes - target_volumes) ** 2) < 0.1

    return ResultContainer(success=success, score=1.0)
