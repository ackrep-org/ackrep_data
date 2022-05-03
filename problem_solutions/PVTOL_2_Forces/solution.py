#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import symbtools as st
from scipy.integrate import solve_ivp
import sympy as sp
import os
import numpy as np

class SolutionData:
    pass


def solve(problem_spec):
    xx_res = solve_ivp(problem_spec.rhs, [problem_spec.tt[0], problem_spec.tt[-1]],
                                     problem_spec.xx0, method='RK45', t_eval=problem_spec.tt)
                                     
    # todo why does problem_spec.tt work

    # print("x1[-1]", xx_res.y[0][-1])
    # print("x2[-1]", xx_res.y[1][-1])
    # print("x3[-1]", xx_res.y[2][-1])
    # print("x4[-1]", xx_res.y[3][-1])
    # print("x5[-1]", xx_res.y[4][-1])
    # print("x6[-1]", xx_res.y[5][-1])


    solution_data = SolutionData()
    solution_data.res = xx_res  # states of system

    save_plot(problem_spec, solution_data)
    return solution_data


def save_plot(problem_spec, solution_data):
    fig1, axs = plt.subplots(nrows=2, ncols=2, figsize=(12.8,9.6))

    # print in axes top left 
    axs[0, 0].plot(solution_data.res.t, np.real(solution_data.res.y[0] ), label = 'x-Position' )
    axs[0, 0].plot(solution_data.res.t, np.real(solution_data.res.y[2] ), label = 'y-Position' )
    axs[0, 0].plot(solution_data.res.t, np.real(solution_data.res.y[4]*180/np.pi ), label = 'angle' )
    axs[0, 0].set_title('Position')
    axs[0, 0].set_ylabel('Position [m]') # y-label Nr 1
    axs[0, 0].set_xlabel('Time [s]') # x-Label für Figure linke Seite
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(solution_data.res.t, solution_data.res.y[1], label = r'$v_x$')
    axs[0, 1].plot(solution_data.res.t, solution_data.res.y[3], label = r'$v_y$')
    axs[0, 1].plot(solution_data.res.t, solution_data.res.y[5]*180/np.pi , label = 'angular velocity')
    axs[0, 1].set_title('Velocities')
    axs[0, 1].set_ylabel('Velocity [m/s]')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].grid()
    axs[0, 1].legend()

    # print in axes bottom left
    uu = problem_spec.uu_func(solution_data.res.t, solution_data.res.y)
    g = 9.81  # m/s^2
    m = 0.25 # kg
    uu = np.array(uu)/(g*m)
    axs[1, 0].plot(solution_data.res.t, uu[0] , label = 'Force left')
    axs[1, 0].plot(solution_data.res.t, uu[1] , label = 'Force right')
    axs[1, 0].set_title('Normalized Input Forces')
    axs[1, 0].set_ylabel(r'Forces normalized to $F_g$') # y-label Nr 1
    axs[1, 0].set_xlabel('Time [s]') # x-Label für Figure linke Seite
    axs[1, 0].grid()
    axs[1, 0].legend()

    # adjust subplot positioning and show the figure
    #fig1.suptitle('', fontsize=16)
    fig1.subplots_adjust(hspace=0.5)

    plt.tight_layout()
    sol_dir = os.path.join(os.path.dirname(__file__), '_solution_data')

    if not os.path.isdir(sol_dir):
        os.mkdir(sol_dir)

    plt.savefig(os.path.join(sol_dir, 'plot.png'), dpi=96 * 2)

    # plt.show()