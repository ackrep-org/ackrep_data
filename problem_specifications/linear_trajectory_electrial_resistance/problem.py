#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""system description: An electrical resistance is considered, which a voltage is applied.
By setting the voltage, the temperature of the resistance can be controlled to a target value.

problem specification for control problem: controller design by using linear trajectory planning.
"""
import numpy as np
import sympy as sp
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
    tt3 = np.linspace(4.5, 5, 50)  # time for evaluation
    poles = [-5]  # desired poles of closed loop
    x0_1 = np.array([55])  # initial conditions for closed loop 1
    x0_2 = np.array([0])  # initial conditions for closed loop 2

    tolerance = 1

    @staticmethod
    def transfer_func():
        s, t, T = sp.symbols("s, t, T")
        # transfer function of the linearized system
        transfer_func = 0.007976 / (s + 0.0001968)
        return transfer_func


def evaluate_solution(solution_data):
    """
    Condition: the temperature of the resistance reaches 310K after 4.5 seconds at the latest
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    success = all(abs(solution_data.y_1[950:] - solution_data.y_func(P.tt3)) < P.tolerance)
    return ResultContainer(success=success, score=1.0)
