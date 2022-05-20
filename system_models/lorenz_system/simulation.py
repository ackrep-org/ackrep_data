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
        print(f"dot_x{i+1} =", eq)

    latt_rhs = lorenz_att.get_rhs_func()

    # Initial State values       
    xx0 = [0.1, 0.1, 0.1]

    t_end = 30
    tt = np.linspace(0, t_end, 10000) # vector of times for simulation
    sol = solve_ivp(latt_rhs, (0, t_end), xx0, t_eval=tt)
    
    save_plot(sol)

    return sol

def save_plot(simulation_data):
    plt.plot(simulation_data.y[0], simulation_data.y[1], label='', lw=1)

    plt.title('x-y Phaseplane')
    plt.xlabel('x',fontsize= 15)
    plt.ylabel('y',fontsize= 15)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(__file__), '_system_model_data')

    plt.savefig(os.path.join(plot_dir, 'plot.png'), dpi=96 * 2)

def evaluate_simulation(simulation_data):
    """
    
    :param simulation_data: simulation_data of system_model
    :return:
    """

    expected_final_state = [-0.522566539750587, -0.830457089853563, 14.033163222999248]
    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.target_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state)
    
    return rc