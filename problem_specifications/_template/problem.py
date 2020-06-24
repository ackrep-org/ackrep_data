""""
This is an example Problem Specification.

Required Interface:

- objects named ProblemData and evaluate_solution must be importable from this module
- evaluate_solution(solution_data) must be callable and return a
  `ackrep_core.ResultContainer`-instance

"""

import numpy as np
from scipy.integrate import odeint

from ackrep_core import ResultContainer

class ProblemSpecification(object):
    """
    DoubleIntegratorTransition
    """
    xx_start = (0, 0)
    xx_end = (1, 0)
    T_transition = 1
    constraints = {"x2": [-3, 3], "u1": [-5, 5]}

    @staticmethod
    def rhs(xx, uu):
        return (xx[1], uu[0])


def evaluate_solution(solution_data):

    # assume solution_data.u_func is callable and returns the desired input trajectory

    P = ProblemSpecification

    def rhs(xx, t):
        u_act = solution_data.u_func(t)
        return P.rhs(xx, u_act)

    umin, umax = P.constraints["u1"]
    x2min, x2max = P.constraints["x2"]

    tt = np.linspace(0, P.T_transition, 1000)

    xx_res = odeint(rhs, P.xx_start, tt)

    # boolean result
    success = abs(xx_res[-1, 1] - P.xx_end) < 1e-2

    success &= all(x2min <= xx_res[:, 1]) and all(xx_res[:, 1] <= x2max)

    uu = solution_data.u_func(tt)
    success &= all(umin <= uu) and all(uu <= umax)

    # a simple score heuristic:
    score = np.clip(1 / solution_data.consumed_time, 0, 10)

    return ResultContainer(success=success, score=score)
