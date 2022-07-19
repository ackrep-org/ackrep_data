#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem solution for control problem: design a controller by using full state feedback.
"""
try:
    import coprime_decomposition as cd  # noqa
except ImportError:
    from method_packages.coprime_decomposition import coprime_decomposition as cd

import sympy as sp
import symbtools as st
import matplotlib.pyplot as plt
import method_trajectory_planning as tp  # noqa
import control
import os
from ackrep_core.system_model_management import save_plot_in_dir

class SolutionData:
    pass


def solve(problem_spec):
    s, t, T = sp.symbols("s, t, T")
    transfer_func = problem_spec.transfer_func()
    n_func, d_func = transfer_func.expand().as_numer_denom()  # separate numerator and denominator
    n_coeffs = [float(c) for c in st.coeffs(n_func, s)]  # coefficients of numerator
    d_coeffs = [float(c) for c in st.coeffs(d_func, s)]  # coefficients of denominator

    b_0 = n_func.coeff(s, 0)
    # Boundary conditions for q and its derivative
    q_a = [problem_spec.YA[0] / b_0, 0, 0, 0]
    q_e = [problem_spec.YB[0] / b_0, 0, 0, 0]

    # generate trajectory of q(t)
    planer = tp.Trajectory_Planning(q_a, q_e, problem_spec.t0, problem_spec.tf, problem_spec.tt)
    planer.dem = d_func
    planer.num = n_func
    q_poly = planer.calc_trajectory()

    # trajectory of input and output
    u_poly, y_poly = planer.num_den_laplace(q_poly[0])

    q_func = st.expr_to_func(t, q_poly[0])
    u_func = st.expr_to_func(t, u_poly)  # desired input trajectory function
    y_func = st.expr_to_func(t, y_poly)  # desired output trajectory function

    # tracking controller
    # numerator and denominator of controller
    cd_res = cd.coprime_decomposition(n_func, d_func, problem_spec.pol)
    tf_k = (cd_res.f_func * n_func) / (cd_res.h_func * d_func)  # open loop
    z_o, n_o = sp.simplify(tf_k).expand().as_numer_denom()
    n_coeffs_o = [float(c) for c in st.coeffs(z_o, s)]  # coefficients of numerator of open loop
    d_coeffs_o = [float(c) for c in st.coeffs(n_o, s)]  # coefficients of denominator open loop

    # in order to make the transfer function to a transfer function object for simulation
    n_coeffs_c = [float(c) for c in st.coeffs(cd_res.f_func, s)]  # coefficients of numerator
    d_coeffs_c = [float(c) for c in st.coeffs(cd_res.h_func, s)]  # coefficients of denominator

    # In order to simulate the closed loop system with the PID controller,
    # the system is divided into two subsystems. one of them with the y_ref as input
    # and the other with u_ref
    close_loop_1 = control.feedback(control.tf(n_coeffs_o, d_coeffs_o))
    close_loop_2 = control.feedback(control.tf(n_coeffs, d_coeffs),
                                    control.tf(n_coeffs_c, d_coeffs_c))

    # subsystem 1 with y_ref
    y_1 = control.forced_response(close_loop_1, problem_spec.tt2, y_func(problem_spec.tt2),
                                  problem_spec.x0_1)
    # subsystem 2 with u_ref
    y_2 = control.forced_response(close_loop_2, problem_spec.tt2, u_func(problem_spec.tt2),
                                  problem_spec.x0_2)

    solution_data = SolutionData()
    solution_data.u = u_func
    solution_data.q = q_func
    solution_data.yy = y_1[1] + y_2[1]
    solution_data.y_func = y_func

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    plt.figure(1)  # simulated trajectory of CuZn-ball
    plt.plot(problem_spec.tt2, solution_data.yy, label='actual trajectory')
    plt.plot(problem_spec.tt1, solution_data.y_func(problem_spec.tt1), ":", label='desired full transition')
    plt.plot(problem_spec.tt, solution_data.y_func(problem_spec.tt), label='desired state transition')
    plt.plot(0, 0.042, 'rx', label='controller switch in')
    plt.xlabel('time [s]')
    plt.ylabel('position [m]')
    plt.title('position of CuZn-ball')
    plt.legend(loc=1)
    
    # save image
    save_plot_in_dir()