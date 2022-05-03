#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import symbtools as st
from scipy.integrate import solve_ivp
import sympy as sp
import os


class SolutionData:
    pass


def solve(problem_spec):
    xx_res = solve_ivp(problem_spec.rhs, [problem_spec.tt[0], problem_spec.tt[-1]],
                                     problem_spec.xx0, method='RK45', t_eval=problem_spec.tt)
                                     
    solution_data = SolutionData()
    solution_data.res = xx_res  # states of system

    save_plot(problem_spec, solution_data)
    return solution_data


def save_plot(problem_spec, solution_data):
    plt.plot(solution_data.res.y[0], solution_data.res.y[1], label='', lw=1)

    plt.title('x-y Phaseplane')
    plt.xlabel('x',fontsize= 15)
    plt.ylabel('y',fontsize= 15)
    plt.legend()
    plt.grid()
  
    plt.tight_layout()
    sol_dir = os.path.join(os.path.dirname(__file__), '_solution_data')

    if not os.path.isdir(sol_dir):
        os.mkdir(sol_dir)

    plt.savefig(os.path.join(sol_dir, 'plot.png'), dpi=96 * 2)

