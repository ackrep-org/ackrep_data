import numpy as np
from scipy.integrate import odeint

from ackrep_core import ResultContainer


class ProblemSpecification(object):
    """
    simple addition
    """
    x = 4
    y = 5


def evaluate_solution(solution_data):


    P = ProblemSpecification


    success = (solution_data == 20)


    return ResultContainer(success=success, score=1.0)
