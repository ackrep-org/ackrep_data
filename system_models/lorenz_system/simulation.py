# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:06:37 2021

@author: Rocky
"""

import numpy as np
import system_model as lac
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
import matplotlib.pyplot as plt
import os

def simulate():
    # Defining Input functions
    lorenz_att = lac.Model()

    rhs_xx_pp_symb = lorenz_att.get_rhs_symbolic()
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i} =", eq)

    latt_rhs = lorenz_att.get_rhs_func()

    # Initial State values       
    xx0 = [0.1, 0.1, 0.1]


    t_end = 30
    tt = np.linspace(0, t_end, 10000) # vector of times for simulation
    sol = solve_ivp(latt_rhs, (0, t_end), xx0, t_eval=tt)
    

    plt.plot(sol.y[0], sol.y[1], label='', lw=1)

    plt.title('x-y Phaseplane')
    plt.xlabel('x',fontsize= 15)
    plt.ylabel('y',fontsize= 15)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.show()

    sol_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(sol_dir, 'plot.png'), dpi=96 * 2)

    return sol

def evaluate_simulation(simulation_data):
    """
    
    :param solution_data: solution data of problem of solution
    :return:
    """

    target_states = [-0.522566539750587, -0.830457089853563, 14.033163222999248]
    success = all(abs(simulation_data.y[i][-1] - target_states[i]) < 1e-2 for i in np.arange(0,3))
    return ResultContainer(success=success, score=1.0)