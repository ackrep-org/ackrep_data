#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system description: a two-body floating system is considered. A magnetic force generated by a current
is applied to a iron ball, which is located directly under the magnet. A CuZn ball below is attached
to that iron ball by a spring with the spring constant kf.

problem specification for control problem:: design of a full state observer to estimate all states of the system.
"""
import numpy as np
import sympy as sp
from ackrep_core import ResultContainer

from system_models.two_mass_floating_bodies_system.system_model import Model


j = 1j  # imaginary unit


class ProblemSpecification(object):
    # system symbols for setting up the equation of motion
    model = Model()
    x1, x2, x3, x4 = model.xx_symb
    xx = sp.Matrix(model.xx_symb)  # states of system
    u1 = model.uu_symb[0] # input of system
    u = [u1]

    # equilibrium points for linearization of the nonlinear system
    eqrt = [(x1, 0.01), (x2, 0.049), (x3, 0), (x4, 0), (u1, 5)]

    '''the first four initial states are the initial condition of nonlinear System,
    The others belong to the small-signal model (observer)'''
    xx0 = np.array([0.02, 0.05, 0, 0, 0.015, 0.005, 0, 0])  # initial condition
    tt = np.linspace(0, 10, 1000)  # vector for the time axis for simulating
    yr = 0  # reference output

    # desired poles of closed system
    poles_cl = [-200, -100, -1 + 16 * j, -1 - 16 * j]

    # desired poles of observer
    poles_o = [-120, -150, -1 + 16 * j, -1 - 16 * j]

    # plotting parameters
    titles_state = ['x1', 'x2', 'x1_dot', 'x2_dot']
    titles_output = ['y']
    x_label = 'time [s]'
    y_label_state = ['position [m]', 'position [m]', 'velocity [m/s]', 'velocity [m/s]']
    y_label_output = ['position [m]']
    graph_color = 'r'
    row_number = 2  # the number of images in each row

    @classmethod
    def rhs(cls):
        """ Right hand side of the equation of motion in nonlinear state space form
        :return:     nonlinear state space
        """        
        return sp.Matrix(cls.model.get_rhs_symbolic_num_params())

    @classmethod
    def output_func(cls):
        """ output equation of the system
        :param xx:   system states
        :param uu:   system input (not used in this case)
        :return:     output equation y = x1
        """
        x1, x2, x3, x4 = cls.xx
        u = cls.u

        return sp.Matrix([x1])


def evaluate_solution(solution_data):
    """
    Condition: all estimated states correspond to the true states after 3 seconds at the latest
    :return:
    """
    res_eva = []
    for i in range(4):
        res_eva.append(all(abs(solution_data.res[600:, i] - solution_data.res[600:, i + 4] < 1e-2)))
    success = all(res_eva)
    return ResultContainer(success=success, score=1.0)
