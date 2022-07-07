#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem solution for control problem: design a controller by using full state feedback.
"""
try:
    import method_full_state_feedback as ftf  # noqa
    import method_system_property as msp  # noqa
except ImportError:
    from method_packages.method_full_state_feedback import method_full_state_feedback as ftf
    from method_packages.method_system_property import method_system_property as msp

import symbtools as st
from scipy.integrate import odeint
import sympy as sp
import matplotlib.pyplot as plt
import os
from ackrep_core.system_model_management import save_plot_in_dir

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
    sim_mod = st.SimulationModel(f, g, xx)
    rhs_eq = sim_mod.create_simfunction(controller_function=controller_func)

    return rhs_eq


def solve(problem_spec):
    """ solution of full state feedback
    :param problem_spec: ProblemSpecification object
    :return: solution_data: states and output values of the stabilized system, and controller gain
    """
    sys_f_body = msp.System_Property()  # instance of the class System_Property
    sys_f_body.sys_state = problem_spec.xx  # state of the system
    sys_f_body.tau = problem_spec.u  # inputs of the system

    # original nonlinear system functions
    sys_f_body.n_state_func = problem_spec.rhs()

    # original output functions
    sys_f_body.n_out_func = problem_spec.output_func()
    sys_f_body.eqlbr = problem_spec.eqrt  # equilibrium point

    # linearize nonlinear system around the chosen equilibrium point
    sys_f_body.sys_linerazition()
    tuple_system = (sys_f_body.aa, sys_f_body.bb, sys_f_body.cc, sys_f_body.dd)  # system tuple

    # calculate controller function
    # sfb for dataclass StateFeedbackResult
    sfb = ftf.state_feedback(tuple_system, problem_spec.poles_cl, problem_spec.xx, problem_spec.eqrt,
                             problem_spec.yr, debug=False)

    # simulation original nonlinear system with controller
    f = sys_f_body.n_state_func.subs(st.zip0(sys_f_body.tau))  # x_dot = f(x) + g(x) * u
    g = sys_f_body.n_state_func.jacobian(sys_f_body.tau)

    rhs = rhs_for_simulation(f, g, problem_spec.xx, sfb.input_func)
    res = odeint(rhs, problem_spec.xx0, problem_spec.tt)

    output_function = sp.lambdify(problem_spec.xx, sys_f_body.n_out_func, modules='numpy')
    yy = output_function(*res.T)

    solution_data = SolutionData()
    solution_data.res = res  # states of system
    solution_data.pre_filter = sfb.pre_filter  # pre-filter
    solution_data.ff = sfb.state_feedback  # controller gain
    solution_data.input_fun = sfb.input_func  # controller function
    solution_data.yy = yy[0][0]  # system output

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    # plotting of the system states
    titles1 = problem_spec.titles_state
    plt.figure(1)
    for i in range(len(titles1)):
        plt.subplot(problem_spec.row_number, int(len(titles1) / problem_spec.row_number), i + 1)
        plt.plot(problem_spec.tt, solution_data.res[:, i], color=problem_spec.graph_color, linewidth=1)
        plt.grid(1)
        plt.title(titles1[i])
        plt.xlabel(problem_spec.x_label)
        plt.ylabel(problem_spec.y_label_state[i])
    plt.tight_layout()

    titles2 = problem_spec.titles_output
    plt.figure(2)
    for i in range(len(titles2)):
        plt.plot(problem_spec.tt, solution_data.yy)
        plt.grid(1)
        plt.title(titles2[i])
        plt.xlabel(problem_spec.x_label)
        plt.ylabel(problem_spec.y_label_output[i])
        plt.tight_layout()
    
    # save image
    save_plot_in_dir(os.path.dirname(__file__))
