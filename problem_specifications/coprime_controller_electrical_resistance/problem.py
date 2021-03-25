#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""system description: An electrical resistance is considered, which a voltage is applied.
By setting the voltage, the temperature of the resistance can be controlled to a target value.

problem specification for control problem: controller design by using the coprime decomposition
to control the temperature of the resistance.
"""
import numpy as np
import sympy as sp
from ackrep_core import ResultContainer


class ProblemSpecification(object):
    t0 = 0  # start time in s
    tf = 5  # final time in s
    tt = np.linspace(0, 10, 1000)  # time axis for simulation
    yr = 310  # target temperature
    pol = [-2]  # desired poles of closed loop
    x0_1 = np.array([280 / 2])  # initial condition for closed loop
    tolerance = 1e-1

    # plotting parameters
    titles_state = ['temperature of resistance']
    titles_output = ['temperature of resistance']
    x_label = ['time [s]']
    y_label_state = ['temperature [k]']
    y_label_output = ['temperature [k]']
    graph_color = 'r'
    row_number = 1  # the number of images in each row

    @staticmethod
    def transfer_func():
        s, t, T = sp.symbols("s, t, T")
        # transfer function of the linearized system
        transfer_func = 0.007976 / (s + 0.0001968)
        return transfer_func


def evaluate_solution(solution_data):
    """
    Condition: the temperature of the resistance reaches 310K after 8 seconds at the latest
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    success = all(abs(solution_data.yy[800:] - [P.yr] * 200) < P.tolerance)
    return ResultContainer(success=success, score=1.0)
