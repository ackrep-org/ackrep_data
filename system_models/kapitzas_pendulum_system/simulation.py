# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:06:37 2021

@author: Rocky
"""

import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
import matplotlib.pyplot as plt
import os

def simulate():
    model = system_model.Model()

    rhs_xx_pp_symb = model.get_rhs_symbolic()
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i+1} =", eq)

    rhs = model.get_rhs_func()


    # Initial State values  
    xx0 = [200/360*2*np.pi,0]

    t_end = 5
    tt = times = np.linspace(0,t_end,10000) 
    sim = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

    uu = model.uu_func(sim.t, xx0)[0] *0.005 +180
    sim.uu = uu

    
    save_plot(sim)

    return sim

def save_plot(sim):
   
    
    # create figure + 2x2 axes array
    fig1, axs = plt.subplots(nrows=2, ncols=1, figsize=(12.8,9.6))

    # print in axes top left
    axs[0].plot(sim.t, np.real(sim.y[0]*360/(2*np.pi)), label = 'Phi')
    #axs[0].plot(sol.t, np.real(sol1.y[0]*360/(2*np.pi)), label = 'Phi_old')
    axs[0].plot(sim.t, list(sim.uu), label ='periodic excitation cos(omega*t)')
    #axs[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    #axs[0].set_yticklabels([r'-$\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    axs[0].set_ylabel('Angle[rad]') # y-label
    axs[0].set_xlabel('Time[s]') # x-label
    axs[0].grid()
    axs[0].legend()

    # print in axes top right 
    axs[1].plot(sim.t, np.real(sim.y[1] ), label = 'Phi_dot')
    axs[1].set_ylabel('Angular velocity[rad/s]') # y-label
    axs[1].set_xlabel('Time[s]') # x-Label
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()

    ## static
    plot_dir = os.path.join(os.path.dirname(__file__), '_system_model_data')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'plot.png'), dpi=96 * 2)

def evaluate_simulation(simulation_data):
    """
    
    :param simulation_data: simulation_data of system_model
    :return:
    """
    
    expected_final_state = [3.1206479177994066, 0.22532090507016805]

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc