import numpy as np
from scipy.integrate import odeint


def solve(problem_spec):

    solution_data = problem_spec.x + problem_spec.y

    return solution_data
