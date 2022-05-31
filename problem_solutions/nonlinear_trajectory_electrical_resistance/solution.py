#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem solution for control problem: design a controller by using nonlinear trajectory planning.
"""
import sympy as sp
import symbtools as st
import matplotlib.pyplot as plt
import method_trajectory_planning as tp  # noqa
import os

from scipy.integrate import odeint


class SolutionData:
    pass


def rhs_for_simulation(f, g, xx, controller_func):
    """
    # calculate right hand side equation for simulation of the nonlinear system
    :param f: vector field
    :param g: input matrix
    :param xx: states of the system
    :param controller_func: input equation (trajectory)
    :return: rhs: equation that is solved
    """

    # call the class 'SimulationModel' to build the
    # 'right hand side'equation for ode
    # sim_mod = st.SimulationModel(f, g, xx)
    sim_mod = st.SimulationModel(f, g, xx)
    rhs_eq = sim_mod.create_simfunction(controller_function=controller_func)

    return rhs_eq


def solve(problem_spec):
    s, t, T = sp.symbols("s, t, T")

    planer = tp.Trajectory_Planning(problem_spec.YA, problem_spec.YB,
                                    problem_spec.t0, problem_spec.tf, problem_spec.tt)
    mod = problem_spec.rhs(problem_spec.xx, problem_spec.uu)
    planer.mod = mod
    planer.yy = problem_spec.output_func(problem_spec.xx, problem_spec.uu)
    planer.ff = mod.f  # xd = f(x) + g(x)*u
    planer.gg = mod.g
    yy = planer.cal_li_derivative()
    ploy_tem = planer.calc_trajectory()
    tem_func = st.expr_to_func(t, ploy_tem[0])

    # tracking controller
    tracking_controller = tp.Tracking_controller(yy, mod.xx, problem_spec.uu, problem_spec.pol, ploy_tem)
    control_law = tracking_controller.error_dynamics()[0]  # control law

    rhs = rhs_for_simulation(planer.ff, planer.gg, mod.xx, control_law)

    res = odeint(rhs, problem_spec.xx0, problem_spec.tt2)
    solution_data = SolutionData()
    solution_data.res = res
    solution_data.p2_func = tem_func

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    # plotting of the system states
    titles = problem_spec.titles_state
    plt.figure(1)
    for i in range(len(titles)):
        plt.subplot(problem_spec.row_number, int(len(titles) / problem_spec.row_number), i + 1)
        plt.plot(problem_spec.tt, solution_data.p2_func(problem_spec.tt), label='desired state transition')
        plt.plot(problem_spec.tt1, solution_data.p2_func(problem_spec.tt1), ":", label='desired full transition')
        plt.plot(problem_spec.tt2, solution_data.res[:, 0], label='actual trajectory')
        plt.plot(0, 275, 'rx', label='controller switch in')
        plt.legend(loc=2)
        plt.title(titles[i])
        plt.xlabel(problem_spec.x_label)
        plt.ylabel(problem_spec.y_label_state[i])
    plt.tight_layout()
    
    # save image
    sol_dir = os.path.join(os.path.dirname(__file__), '_solution_data')

    if not os.path.isdir(sol_dir):
        os.mkdir(sol_dir)

    plt.savefig(os.path.join(sol_dir, 'plot.png'), dpi=96*2)
