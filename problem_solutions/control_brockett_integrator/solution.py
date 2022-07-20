#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem solution for control problem: brokett-integrator does not admit a continuously differentiable control
law
"""

try:
    import method_geometry_inspired_control as mgic  # noqa
except ImportError:
    from method_packages.method_geometry_inspired_control import method_geometry_inspired_control as mgic

import matplotlib.pyplot as plt
import symbtools as st
from scipy.integrate import odeint
import sympy as sp
import os
import numpy as np
from ackrep_core.system_model_management import save_plot_in_dir

from ipydex import IPS


class SolutionData:
    pass


def rhs_for_simulation(f, g, xx, controller_func):
    """
    calculate right hand side equation for simulation of the nonlinear system
    :param f: vector field
    :param g: input matrix
    :param xx: states of the system
    :param controller_func: input equation (trajectory)
    :return: rhs: equation that is solved
    """

    # call the class 'SimulationModel' to build the
    # 'right hand side' equation for ode
    sim_mod = st.SimulationModel(f, g, xx)
    rhs_eq = sim_mod.create_simfunction(controller_function=controller_func)

    return rhs_eq


def solve(problem_spec):
    """ 
    calculate the solution data using the method package method_geometry_inspired_control
    :param problem_spec: ProblemSpecification object
    :return: solution_data: list of arrays of the state histories
    """
    
    x1, x2, x3 = xx = problem_spec.model.xx_symb
    v1, v2 = vv = problem_spec.vv    
    uue = mgic.cylindical_coordinates(problem_spec)
    r1_opt_interp = mgic.shortest_curve(problem_spec)
    # function to calculate u1, u2, from v1, v2
    vv_to_uu = sp.lambdify((v1, v2, x1, x2), list(uue))

    z_tol = 1e-2
    r_tol = 1e-2
    Tend = 6
    dt = .005

    var_controller = [r1_opt_interp, z_tol, r_tol, vv_to_uu]

    N = 20
    np.random.seed(3)

    # initial values: x1, x2 uniformly random in (-1, 1)
    # x3 in (0, 1)
    xx0_values = np.column_stack( ((np.random.rand(N, 2) - .5), np.random.rand(N, 1)))

    res = [mgic.simulate(var_controller, xx0, Tend, dt) for xx0 in xx0_values]

    tt = np.arange(0, Tend+2*dt, dt)[:res[0].shape[0]]

    save_plot(res, tt)

    return res


def save_plot(solution_data, tt):
    """
    plot and save results of the solution
    ond 3D-plot and five 2D-plots

    :param solution_data: list of simulated data
    :param tt: vector of times
    :return: None
    """

    # swap some of the results, such that 3d view looks nicer
    solution_data[0], solution_data[1] = solution_data[1], solution_data[0]
        
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['figure.subplot.bottom'] = .2

    plt.rcParams['figure.subplot.left'] = .05
    plt.rcParams['figure.subplot.right'] = .9


    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(4, 1, 1, projection='3d')
    # plot axis

    kwargs1 = dict(color="0.3", ls="solid", lw=2)

    # plot coordinate system
    ax.plot([-.1, .65], [0, 0], [0, 0], **kwargs1)
    ax.plot([0, 0], [-.1, .65], [0, 0], **kwargs1)
    ax.plot([0, 0], [0, 0], [-.1, 1], **kwargs1)

    ax.set_xlabel("$x_1$")
    ax.set_xticks([-.6, .6])
    ax.set_ylabel("$x_2$")
    ax.set_yticks([-.6, .6])
    ax.set_zlabel("$x_3$")
    ax.set_zticks([0, 1])
    for i, xxn in enumerate(solution_data):
        ax.plot(xxn[:, 0], xxn[:, 1], xxn[:, 2])


    ax = fig.add_subplot(4, 2, 3)
    ax.set_ylabel("$x_1$")
    ax.set_xlabel("$t$")
    for i, xxn in enumerate(solution_data):
        z = xxn[:, 2]   
        ax.plot(tt, xxn[:, 0])

    ax = fig.add_subplot(4, 2, 5)
    ax.set_ylabel("$x_2$")
    ax.set_xlabel("$t$")
    for i, xxn in enumerate(solution_data):
        ax.plot(tt, xxn[:, 1])

    ax = fig.add_subplot(4, 2, 7)
    ax.set_ylabel("$x_3$")
    ax.set_xlabel("$t$")
    for i, xxn in enumerate(solution_data):
        ax.plot(tt, xxn[:, 2])

    ax = fig.add_subplot(4, 2, 6)
    ax.set_ylabel("$r$")
    ax.set_xlabel("$t$")
    for i, xxn in enumerate(solution_data):
        r = np.sqrt(xxn[:, 0]**2 + xxn[:, 1]**2)
        ax.plot(tt, r)

    ax = fig.add_subplot(4, 2, 8)
    ax.set_ylabel(r"$\varphi$")
    ax.set_xlabel("$t$")
    for i, xxn in enumerate(solution_data):
        phi_ = np.arctan2(xxn[:, 0], xxn[:, 1])
        phi = mgic.cont_continuation(phi_)
        ax.plot(tt, phi)
    

    plt.tight_layout()
    
    # save image
    save_plot_in_dir()
