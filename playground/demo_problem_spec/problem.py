import numpy as np
from scipy.integrate import odeint

from ackrep_core import ResultContainer


class ProblemSpecification(object):
    """
    simple addition
    """
    x = 10
    y = -3


def evaluate_solution(solution_data):


    P = ProblemSpecification


    success = (solution_data == 7)


    return ResultContainer(success=success, score=1.0)
