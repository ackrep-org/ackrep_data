#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem solution for control problem: controller design by using
coprime decomposition.
"""
try:
    import coprime_decomposition as cd  # noqa
except ImportError:
    from method_packages.coprime_decomposition import coprime_decomposition as cd

import sympy as sp
import symbtools as st
import matplotlib.pyplot as plt
import control
import os
from ackrep_core.system_model_management import save_plot_in_dir


class SolutionData:
    pass


def solve(problem_spec):
    """ solution of coprime decomposition
    :param problem_spec: ProblemSpecification object
    :return: solution_data: output value of the system and controller function
    """
    s, t, T = sp.symbols("s, t, T")
    transfer_func = problem_spec.transfer_func()
    z_func, n_func = transfer_func.expand().as_numer_denom()  # separate numerator and denominator

    # tracking controller
    # numerator and denominator of controller
    cd_res = cd.coprime_decomposition(z_func, n_func, problem_spec.pol)
    tf_k = (cd_res.f_func * z_func) / (cd_res.h_func * n_func)  # open_loop
    z_o, n_o = sp.simplify(tf_k).expand().as_numer_denom()
    n_coeffs_o = [float(c) for c in st.coeffs(z_o, s)]  # coefficients of numerator of open_loop
    d_coeffs_o = [float(c) for c in st.coeffs(n_o, s)]  # coefficients of denominator of open_loop

    # feedback
    close_loop = control.feedback(control.tf(n_coeffs_o, d_coeffs_o))

    # simulate system with controller with initial error (190K instead of 200K).
    y = control.forced_response(close_loop, problem_spec.tt, problem_spec.yr, problem_spec.x0_1)

    solution_data = SolutionData()
    solution_data.yy = y[1]
    solution_data.controller_n = cd_res.f_func
    solution_data.controller_d = cd_res.h_func
    solution_data.controller_ceoffs = cd_res.c_coeffs

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    titles = problem_spec.titles_state
    plt.figure(1)  # plotting
    for i in range(len(titles)):
        plt.plot(problem_spec.tt, solution_data.yy)
        plt.grid(1)
        plt.title(titles[i])
        plt.xlabel(problem_spec.x_label[i])
        plt.ylabel(problem_spec.y_label_state[i])
    plt.tight_layout()

    # save image
    save_plot_in_dir()

