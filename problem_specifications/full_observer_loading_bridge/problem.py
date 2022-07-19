#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
system description: A loading bridge is considered, which consists of a wagon with the mass M,
a rope with the constant length l, which is attached to the wagon, and a load,
which is located at the free end of the rope. The force that can be applied to the wagon
is available as a manipulated variable.

problem specification for control problem: design of a full observer to estimate all states of the system.
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
    xx0 = np.array([0.2, 0.5, 0.2, 0.1, 0, 0, 0, 0])  # initial condition
    tt = np.linspace(0, 5, 1000)  # vector for the time axis for simulating
    poles_cl = [-3, -3, -3, -3]  # desired poles for closed loop
    poles_o = [-10, -10, -6, -6]  # poles of the observer dynamics
    yr = 0  # target value pf output

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
        """ output equation of the system
        :return:     output equation y = x1
        """
        x1, x2, x3, x4 = cls.xx
        u = cls.u
        l = cls.model.pp_str_dict["l"]

        return sp.Matrix([x1 + l * sin(x2)])


def evaluate_solution(solution_data):
    """
    Condition: all estimated states correspond to the true states after 4 seconds at the latest
    :return:
    """
    res_eva = []
    for i in range(4):
        res_eva.append(all(abs(solution_data.res[400:, i] - solution_data.res[400:, i + 4] < 1e-2)))
    success = all(res_eva)
    return ResultContainer(success=success, score=1.0)
