#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sympy import cos, sin
from math import pi
from ackrep_core import ResultContainer


class ProblemSpecification(object):
    # system symbols for setting up the equation of motion
    p1, p2, pdot1, pdot2 = sp.symbols("p1, p2, pdot1, pdot2")
    F = sp.Symbol('F')  # nonlinear external magnetic force on the body
    xx = sp.Matrix([p1, p2, pdot1, pdot2])  # states of system
    u = [F]  # input of system

    # equilibrium point
    eqrt = [(p1, 0), (p2, 0), (pdot1, 0), (pdot2, 0), (F, 0)]
    xx0 = np.array([0.2, pi / 6, 0.5, 0.2])  # initial condition
    yr = 0.5  # reference position
    tt = np.linspace(0, 8, 1000)  # vector for the time axis for simulating
    q = np.diag([15, 15, 12, 13])  # desired poles
    r = np.diag([1])  # initial condition

    @staticmethod
    def rhs(xx, uu):
        """ Right hand side of the equation of motion in nonlinear state space form
        :param xx:   system states
        :param uu:   system input
        :return:     nonlinear state space
        """
        m1 = 1  # mass of the iron ball in kg
        m2 = 0.25  # mass of the brass ball in kg
        g = 9.81  # acceleration of gravity in m/s^2
        l = 1  # geometry constant

        x1, x2, x3, x4 = xx

        # motion of equations
        p1_dot = (uu[0] + (g * m2 * sin(2 * x2)) / 2 + l * m2 * x4 ** 2 * sin(x2)) / (m1 + m2 * (sin(x2) ** 2))
        p2_dot = - (g * (m1 + m2) * sin(x2) + (uu[0] + l * m2 * x4 ** 2 * sin(x2))
                    * cos(x2)) / (l * (m1 + m2 * (sin(x2) ** 2)))

        ff = sp.Matrix([x3,
                        x4,
                        p1_dot,
                        p2_dot])

        return ff

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
