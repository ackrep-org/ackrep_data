#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system description: A loading bridge is considered, which consists of a wagon with the mass M,
a rope with the constant length l, which is attached to the wagon, and a load,
which is located at the free end of the rope. The force that can be applied to the wagon
is available as a manipulated variable.

problem specification for control problem: design of a full state feedback controller to control
the x-position of the load to 1.5m.
"""

import numpy as np
import sympy as sp
from sympy import cos, sin
from math import pi
from ackrep_core import ResultContainer
from system_models.loading_bridge_system.system_model import Model


class ProblemSpecification(object):
    # system symbols for setting up the equation of motion
    model = Model()
    x1, x2, x3, x4 = model.xx_symb
    xx = sp.Matrix(model.xx_symb)  # states of system
    u = [model.uu_symb[0]] # input of system

    # equilibrium point for linearization of the nonlinear system
    eqrt = [(x1, 0), (x2, 0), (x3, 0), (x4, 0), (u, 0)]
    xx0 = np.array([0.2, pi/6, 1, 0.2])  # initial condition for simulation
    tt = np.linspace(0, 5, 1000)  # vector for the time axis for simulating
    poles_cl = [-3, -3, -3, -3]  # desired poles of closed loop
    yr = 1.5  # reference output

    # plotting parameters
    titles_state = ['x1', 'x2', 'x1_dot', 'x2_dot']
    titles_output = ['y']
    x_label = 'time [s]'
    y_label_state = ['position [m]', 'angular position [rad]', 'velocity [m/s]', 'angular velocity [rad/s]']
    y_label_output = ['x-position of pendulum m']
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
        """ output equation of the system: x-position of the load
        :return:     output equation y = x1
        """
        x1, x2, x3, x4 = cls.xx
        u = cls.u
        l = cls.model.pp_str_dict["l"]  # geometry constant

        return sp.Matrix([x1 + l * sin(x2)])


def evaluate_solution(solution_data):
    """
    Condition: the x-position of the load reaches 1.5m after at least 4 seconds
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    success = all(abs(solution_data.yy[800:] - [P.yr] * 200) < 1e-2)
    return ResultContainer(success=success, score=1.0)


