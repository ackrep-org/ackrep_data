# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:06:37 2021

@author: Rocky
"""

import numpy as np
import system_model as bi
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
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

    ## static
    plot_dir = os.path.join(os.path.dirname(__file__), '_system_model_data')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    pyplot.savefig(os.path.join(plot_dir, 'plot.png'), dpi=96 * 2)

def evaluate_simulation(simulation_data):
    """

    :param simulation_data: simulation_data of system_model
    :return:
    """
    
    target_states = [1.5126199035042642e-05, -1.6950186169609194e-05, 0.7956450415081588]
    
    rc = ResultContainer(score=1.0)
    rc.target_state_errors = [simulation_data.y[i][-1] - target_states[i] for i in np.arange(0, len(simulation_data.y))]
    rc.success = all(abs(np.array(rc.target_state_errors)) < 1e-2)
    
    return rc