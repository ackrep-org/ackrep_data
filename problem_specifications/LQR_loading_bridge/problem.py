 
"""
system description: A loading bridge is considered, which consists of a wagon with the mass M,
a rope with the constant length l, which is attached to the wagon, and a load,
which is located at the free end of the rope. The force that can be impressed on the wagon
is available as a manipulated variable.

problem specification for control problem: design of the LQR controller to to control and
stabilize the x-position of the load.
"""
import numpy as np
import sympy as sp
from sympy import cos, sin, symbols
from math import pi
from ackrep_core import ResultContainer
from system_models.loading_bridge_system.system_model import Model

from ipydex import IPS

class ProblemSpecification(object):
    # system symbols for setting up the equation of motion
    model = Model()
    x1, x2, x3, x4 = model.xx_symb
    xx = sp.Matrix(model.xx_symb)  # states of system
    u = [model.uu_symb[0]] # input of system

    # equilibrium point
    eqrt = [(x1, 0), (x2, 0), (x3, 0), (x4, 0), (u, 0)]
    xx0 = np.array([0.2, pi / 6, 0.5, 0.2])  # initial condition
    yr = 0.5  # reference position
    tt = np.linspace(0, 8, 1000)  # vector for the time axis for simulating
    q = np.diag([15, 15, 12, 13])  # state weights matrix
    r = np.diag([1])  # input weights matrix



    def rhs(model):
        """ Right hand side of the equation of motion in nonlinear state space form
        :param xx:   system states
        :param uu:   system input
        :return:     nonlinear state space
        """
        
        return sp.Matrix(model.get_rhs_symbolic_num_params())


    @staticmethod
    def output_func(xx, uu):
        """ output equation of the system
        :param xx:   system states
        :param uu:   system input (not used in this case)
        :return:     output equation y = x1
        """
        x1, x2, x3, x4 = xx
        u = uu
        l = 1  # geometry constant

        return sp.Matrix([x1 + l * sin(x2)])


def evaluate_solution(solution_data):
    """
    Condition: the x-position of the load reaches 1.5m after 6 seconds at the latest
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    success = all(abs(solution_data.yy[750:] - [P.yr] * 250) < 1e-2)
    return ResultContainer(success=success, score=1.0)
