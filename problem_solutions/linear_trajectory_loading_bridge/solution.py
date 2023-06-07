#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem solution for control problem: design a controller by using full state feedback.
"""
try:
    import coprime_decomposition as cd  # noqa
except ImportError:
    from method_packages.coprime_decomposition import coprime_decomposition as cd

import symbtools as st
import matplotlib.pyplot as plt
import method_trajectory_planning as tp  # noqa
from pyblocksim import *
import os
from ackrep_core.system_model_management import save_plot_in_dir


class SolutionData:
    pass


def solve(problem_spec):
    s, t, T = sp.symbols("s, t, T")
    transfer_func = problem_spec.transfer_func()
    z_func, n_func = transfer_func.expand().as_numer_denom()  # separate numerator and denominator
    z_coeffs = [float(c) for c in st.coeffs(z_func, s)]  # coefficients of numerator
    n_coeffs = [float(c) for c in st.coeffs(n_func, s)]  # coefficients of denominator

    b_0 = z_func.coeff(s, 0)
    # Boundary conditions for q and its derivative
    q_a = [problem_spec.YA[0] / b_0, 0, 0, 0]
    q_e = [problem_spec.YB[0] / b_0, 0, 0, 0]

    # generate trajectory of q(t)
    planer = tp.Trajectory_Planning(q_a, q_e, problem_spec.t0, problem_spec.tf, problem_spec.tt)
    planer.dem = n_func
    planer.num = z_func
    q_poly = planer.calc_trajectory()

    # trajectory of input and output
    u_poly, y_poly = planer.num_den_laplace(q_poly[0])

    q_func = st.expr_to_func(t, q_poly[0])
    u_func = st.expr_to_func(t, u_poly)  # desired input trajectory function
    y_func = st.expr_to_func(t, y_poly)  # desired output trajectory function

    # tracking controller

    # numerator and denominator of controller
    cd_res = cd.coprime_decomposition(z_func, n_func, problem_spec.pol)
    u1, u2, fb = inputs("u1, u2, fb")  # external force and feedback
    SUM1 = Blockfnc(u1 - fb)
    Controller = TFBlock(cd_res.f_func / cd_res.h_func, SUM1.Y)
    SUM2 = Blockfnc(u2 + Controller.Y)
    System = TFBlock(z_func / n_func, SUM2.Y)
    loop(System.Y, fb)
    t1, states = blocksimulation(6, {u1: y_func, u2: u_func})  # simulate 10 seconds
    t1 = t1.flatten()
    bo = compute_block_ouptputs(states)

    solution_data = SolutionData()
    solution_data.u = u_func
    solution_data.q = q_func
    solution_data.yy = bo[System]
    solution_data.y_func = y_func
    solution_data.tt = t1

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    plt.figure(1)  # simulated trajectory of CuZn-ball
    plt.plot(solution_data.tt, solution_data.yy, label="actual trajectory")
    plt.plot(problem_spec.tt1, solution_data.y_func(problem_spec.tt1), ":", label="desired full transition")
    plt.plot(problem_spec.tt, solution_data.y_func(problem_spec.tt), label="desired state transition")
    plt.plot(0, 0, "rx", label="controller switch in")
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("x-position of pendulum")
    plt.legend(loc=1)

    save_plot_in_dir("plot1.png")

    plt.figure(2)
    plt.plot(problem_spec.tt1, solution_data.u(problem_spec.tt1))
    plt.xlabel("time [s]")
    plt.ylabel("force [N]")
    plt.title("external force")

    plt.tight_layout()

    save_plot_in_dir("plot2.png")
