#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""system description: An electrical resistance is considered to which a voltage is applied.
By setting the voltage, the temperature of the resistance can be controlled to a target value.

problem specification for control problem: design of the LQR controller to control
the temperature of the resistance.
"""
import numpy as np
import sympy as sp
from ackrep_core import ResultContainer


class ProblemSpecification(object):
    # system symbols for setting up the equation of motion
    p1, p1_dot = sp.symbols("p1, p1_dot")  # p1: temperature [k] p2: velocity of temperature [K/t]
    U = sp.Symbol("U")  # input symbol
    xx = sp.Matrix([p1])  # states of system
    u = [U]  # input voltage of system

    # equilibrium points for linearization of the nonlinear system
    # equilibrium point p1 must be at least as high as the environment temperature Ta
    # in this case Ta = 293.15 K
    eqrt = [(p1, 293.15), (U, 2)]
    xx0 = np.array([280])  # initial condition
    yr = 100  # reference temperature (w.r.t to equilibrium)
    tt = np.linspace(0, 10, 1000)  # vector for the time axis for simulation
    q = np.diag([650])  # desired poles
    r = np.diag([0.002])  # initial condition

    @staticmethod
    def rhs(xx, uu):
        """ Right hand side of the equation of motion in nonlinear state space form
        as symbolic expression

        :param xx:   system states
        :param uu:   system input
        :return:     nonlinear state space
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

        x1 = xx[0]
        u = uu[0]

        p1_dot = u / (c * R0 * (1 + alpha * (x1 - Tr))) - (sigma * A * (x1 ** 4 - Ta ** 4)) / c
        ff = sp.Matrix([p1_dot])

        return ff

    @staticmethod
    def output_func(xx, uu):
        """ output equation of the system
        :param xx:   system states
        :param uu:   system input (not used in this case)
        :return:     output equation y = x1
        """
        x1 = xx[0]
        u = uu[0]

        return sp.Matrix([x1])


def evaluate_solution(solution_data):
    """
    Condition: the temperature of the resistance reaches 300K after 9 seconds at the latest
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    yy_lin = solution_data.yy[900:] - 293.15
    success = np.allclose(yy_lin[900:], P.yr,  1e-1)
    return ResultContainer(success=success, score=1.0)
