#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""system description: An electrical resistance is considered, which a voltage is applied.
By setting the voltage, the temperature of the resistance can be controlled to a target value.

problem specification for control problem: controller design by using nonlinear trajectory planning.
"""
import numpy as np
import sympy as sp
from symbtools import modeltools as mt
import symbtools as st
import math
from ackrep_core import ResultContainer

j = 1j  # imaginary unit


class ProblemSpecification(object):
    YA = [280, 0]  # initial conditions of the CuZn-Ball p2 (position in m)
    YB = [310, 0]  # final conditions of the CuZn-Ball p2 (position in m)
    t0 = 0  # start time in s
    tf = 5  # final time in s
    tt = np.linspace(0, 5, 1000)  # period of state transition
    tt1 = np.linspace(-1, 6, 1000)  # period of full transition
    tt2 = np.linspace(0, 6, 1000)  # period of full transition
    poles = [-5]  # desired poles of closed loop
    yr = 310  # target temperature

    xx0 = [275]
    p1 = sp.symbols("p1")  # p1: temperature [k]
    U = sp.Symbol("U")  # input symbol
    xx = [p1]  # states of system
    uu = [U]  # fictive input

    pol = [-5]  # desired poles for error dynamics

    tolerance = 2e-1

    # plotting parameters
    titles_state = ['temperature of resistance']
    titles_output = ['temperature of resistance']
    x_label = ['time [s]']
    y_label_state = ['temperature [k]']
    y_label_output = ['temperature [k]']
    graph_color = 'r'
    row_number = 1  # the number of images in each row


    @staticmethod
    def rhs(xx, uu):
        """ Right hand side of the equation of motion in nonlinear state space form
        :param xx:   system state
        :param uu:   system input
        :return:     nonlinear state function
        """
        R0 = 6  # reference resistance at the reference temperature in [Ohm]
        Tr = 303.15  # reference temperature in [K]
        alpha = 3.93e-3  # temperature coefficient in [1/K]
        Ta = 293.15  # environment temperature in [K]
        sigma = 5.67e-8  # Stefan–Boltzmann constant in [W/m**2/k**4]
        A = 0.0025  # surface area of resistance m ** 2
        c = 87  # heat capacity in [J/k]
        '''
        heat capacity is equal to specific heat capacity * mass of resistance
        specific heat capacity for Cu: 394 [J/kg/K], density of Cu_resistance : 8.86[g/cm**3]
        volume of Cu_resistance: 0.0025 [m**2] * 0.01 [m] 
        '''
        x1 = xx[0]  # state: temperature
        u = uu[0]

        p1_dot = u / (c * R0 * (1 + alpha * (x1 - Tr))) - (sigma * A * (x1 ** 4 - Ta ** 4)) / c

        ff = sp.Matrix([p1_dot])
        mod = mt.SymbolicModel()
        mod.xx = sp.Matrix([x1])
        mod.eqns = ff
        mod.tau = sp.Matrix([u])
        mod.f = ff.subs(st.zip0(mod.tau))
        mod.g = ff.jacobian(mod.tau)
        return mod

    @staticmethod
    def output_func(xx, uu):
        """ output equation of the system
        :param xx:   system states
        :param uu:   system input (not used in this case)
        :return:     output equation y = x1
        """
        u = uu

        return sp.Matrix([xx[0]])


def evaluate_solution(solution_data):
    """
    Condition: the temperature of the resistance reaches 310K after 8 seconds at the latest
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    success = all(abs(solution_data.res[800:][0] - [P.yr] * 200) < P.tolerance)

    return ResultContainer(success=success, score=1.0)
