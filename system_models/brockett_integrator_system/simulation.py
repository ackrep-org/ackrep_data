# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:06:37 2021

@author: Rocky
"""

import numpy as np
import system_model as bi
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as pyplot
import os

def simulate():

    model = bi.Model()

    rhs_xx_pp_symb = model.get_rhs_symbolic()

    print("Simulation with input functions: u1 = sin(omega*t), u2 = cos(omega*t)\n")
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i+1} =", eq)

    rhs = model.get_rhs_func()

    xx0 = [0, 0, 0]
    
    t_end = 10
    tt = np.linspace(0, t_end, 1000) # vector of times for simulation
    sim = solve_ivp(rhs, (0,t_end), xx0, t_eval=tt)
    
    save_plot(sim)

    return sim

def save_plot(sol):
    
    pyplot.plot(sol.t,sol.y[0],label='x1')
    pyplot.plot(sol.t,sol.y[1],label='x2')
    pyplot.plot(sol.t,sol.y[2],label='x3')


    pyplot.title('State progress')
    pyplot.xlabel('Time[s]', fontsize= 15)
    pyplot.legend()
    pyplot.grid()

    pyplot.tight_layout()

    save_plot_in_dir(os.path.dirname(__file__))

def evaluate_simulation(simulation_data):
    """

    :param simulation_data: simulation_data of system_model
    :return:
    """
    
    expected_final_state = [1.5126199035042642e-05, -1.6950186169609194e-05, 0.7956450415081588]
    
    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)
    
    return rc