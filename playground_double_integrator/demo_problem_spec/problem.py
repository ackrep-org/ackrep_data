import numpy as np
from scipy.integrate import odeint

from ackrep_core import ResultContainer

from ipydex import IPS

import matplotlib.pyplot as plt


class ProblemSpecification(object):
    """
    DoubleIntegratorTransition
    """
    xx_start = np.array([0, 0])
    xx_end = np.array([1, 0])
    T_transition = 1
    constraints = {}

    @staticmethod
    def rhs(xx, uu):
        return (xx[1], uu[0])


def evaluate_solution(solution_data):

    # assume solution_data.u_func is callable and returns the desired input trajectory

    P = ProblemSpecification

    def rhs(xx, t):
        u_act = solution_data.u_func(t)
        return P.rhs(xx, u_act)

    tt = np.linspace(0, P.T_transition, 1000)

    xx_res = odeint(rhs, P.xx_start, tt)

    # boolean result
    success = all(abs(xx_res[-1] - P.xx_end) < 1e-2)

    #uu = np.array([solution_data.u_func(t)[0] for t in tt])
    #xx_pyt = np.array([solution_data.x_func(t) for t in tt])

    return ResultContainer(success=success)
