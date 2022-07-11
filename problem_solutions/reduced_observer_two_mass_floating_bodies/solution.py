#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem solution : using reduced observer to estimate states which
not belong to the outputs of the two-mass system
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import method_observer_full_reduced as ofr  # noqa
import method_system_property as msp  # noqa
import os
from ackrep_core.system_model_management import save_plot_in_dir


class SolutionData:
    pass


def solve(problem_spec):
    """ solution of reduced observer
    the design of a linear full observer is based on a linear system.
    therefore the non-linear system should first be linearized at the beginning
    :param problem_spec: ProblemSpecification object
    :return: solution_data: states and output values of the system
    """
    sys_f_body = msp.System_Property()  # instance of the class System_Property
    sys_f_body.sys_state = problem_spec.xx  # state of the system
    sys_f_body.tau = problem_spec.u  # inputs of the system

    # original nonlinear system functions
    sys_f_body.n_state_func = problem_spec.rhs(problem_spec.xx, problem_spec.u)

    # original output functions
    sys_f_body.n_out_func = problem_spec.output_func(problem_spec.xx, problem_spec.u)
    sys_f_body.eqlbr = problem_spec.eqrt  # equilibrium point
    # linearize nonlinear system around the chosen equilibrium point
    sys_f_body.sys_linerazition()

    tuple_system = (sys_f_body.aa, sys_f_body.bb, sys_f_body.cc, sys_f_body.dd)  # system tuple
    yy_f, xx_f, tt_f, sys_f = ofr.reduced_observer(tuple_system, problem_spec.poles_o,
                                                   problem_spec.poles_cl, problem_spec.xx0,
                                                   problem_spec.tt, debug=False)
    solution_data = SolutionData()
    solution_data.yy = yy_f
    solution_data.xx = xx_f

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    titles = ['x1', 'x2', 'x1_dot', 'x2_dot']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(problem_spec.tt, solution_data.xx[:, i], color='k', linewidth=1)
        if i >= 1:
            plt.plot(problem_spec.tt, solution_data.yy[:, i - 1], 'r--', linewidth=1)
        plt.grid(1)
        plt.title(titles[i])
        if i < 2:
            plt.ylabel('position m')
        else:
            plt.ylabel('velocity m/s')
    plt.tight_layout()
    

    # plotting the error between true value and estimated value
    # with initial errors
    titles2 = ['x2', 'x1_dot', 'x2_dot']
    plt.figure(2)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(problem_spec.tt, solution_data.xx[:, i + 1] - solution_data.yy[:, i], color='k', linewidth=1)
        plt.grid(1)
        plt.title(titles2[i])
    plt.tight_layout()
    
    # save image
    save_plot_in_dir(os.path.dirname(__file__), plt)
