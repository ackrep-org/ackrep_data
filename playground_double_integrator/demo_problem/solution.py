"""
This example of the double integrator demonstrates how to pass constraints to PyTrajectory.
"""
# imports
from pytrajectory import TransitionProblem
import numpy as np
from scipy.interpolate import interp1d
from ipydex import IPS


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics

    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)

    :return:        xdot
    """
    x1, x2 = xx
    u1, = uu
    
    ff = [x2,
          u1]
    
    return ff


class SolutionData():
    pass



def solve(problem_spec):
    # system state boundary values for a = 0.0 [s] and b = 2.0 [s]
    xa = problem_spec.xx_start
    xb = problem_spec.xx_end

    T_end = problem_spec.T_transition

    # constraints dictionary
    con = problem_spec.constraints

    # create the trajectory object
    S = TransitionProblem(f, a=0.0, b=T_end, xa=xa, xb=xb, constraints=con, use_chains=False)

    # start
    x, u = S.solve()

    solution_data = SolutionData()
    solution_data.x_func = x
    solution_data.u_func = u

    return solution_data
