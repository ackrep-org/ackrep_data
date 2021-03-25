#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system description: A loading bridge is considered, which consists of a wagon with the mass M,
a rope with the constant length l, which is attached to the wagon, and a load,
which is located at the free end of the rope. The force that can be impressed on the wagon
is available as a manipulated variable.

problem specification for control problem: plan the trajectory of x-position of the load
based on the transfer function.then let the real trajectory of the load converge towards the
target trajectory by using the tracking controller.(x-position of the load from 0m to 1.5m within 5s.)
"""

import sympy as sp
import numpy as np
from ackrep_core import ResultContainer


class ProblemSpecification(object):
    YA = [1, 0, 0, 0]  # initial conditions of x-position of the load (in [m])
    YB = [-2, 0, 0, 0]  # final conditions of x-position of the load (in [m])
    t0 = 0  # start time in s
    tf = 5  # final time in s
    tt = np.linspace(0, 5, 1000)  # period of state transition
    tt1 = np.linspace(-1, 6, 1000)  # period of full transition
    tt2 = np.linspace(0, 6, 1000)  # period for system with controller
    tt3 = np.linspace(4, 6, 400)  # time axis for evaluation

    # pol = [-5, -6, -7, -8]  # desired poles of closed loop
    # x0_1 = np.array([0.5 / 26, 0, 0, 0])  # initial conditions for closed loop 1
    # x0_2 = np.array([0, 0, 0, 0])  # initial conditions for closed loop 2
    pol = [-5, -6, -7, -8, -9, -10, -11]  # desired poles of closed loop
    x0_1 = np.array([0, 0, 0, 0, 0.5/4.13, 0, 0])  # initial conditions for closed loop 1
    x0_2 = np.array([0, 0, 0, 0, 0, 0, 0])  # initial conditions for closed loop 2
    tolerance = 1e-1  # tolerance for evaluation of solution data

    @staticmethod
    def transfer_func():
        s, t, T = sp.symbols("s, t, T")
        # transfer function of the linearized system
        transfer_func = 9.81 / (s ** 4 + 12.26 * s ** 2)
        return transfer_func


def evaluate_solution(solution_data):
    """
    Condition: the x-position of the load corresponds to the desired trajectory after 4 seconds at the latest
    :param solution_data: solution data of problem of solution
    # toleranz:
    :return:
    """
    P = ProblemSpecification
    success = all(abs(solution_data.yy[800:1200] - solution_data.y_func(P.tt3)) < P.tolerance)
    return ResultContainer(success=success, score=1.0)
