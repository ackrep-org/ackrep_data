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
    
    pp1 = [3, 2]
    pp3 = [3, 5, 0.5, 1]
    pp4 = [3, 5, 0.5, 1, 2]
    pp5 = [3, 5, 0.5, 1, 2, 0.02]
    
    model1 = system_model.Model(x_dim=1, pp=pp1)
    model2 = system_model.Model()
    model3 = system_model.Model(x_dim=3, pp=pp3)
    model4 = system_model.Model(x_dim=4, pp=pp4)
    model5 = system_model.Model(x_dim=5, pp=pp5)
    
    model = [model1, model2, model3, model4, model5]

    print('Simulation for PT1 to PT5 with input function u=sin(omega*t) \n')

    for i in range(len(model)):
        rhs_xx_pp_symb = model[i].get_rhs_symbolic()
        print("Computational Equations PT"+str(i+1)+":\n")
        for i, eq in enumerate(rhs_xx_pp_symb):
            print(f"dot_x{i} =", eq, "\n")


    # Initial State values  
    xx0_1 = np.zeros(1)
    xx0_2 = np.zeros(2)
    xx0_3 = np.zeros(3)
    xx0_4 = np.zeros(4)
    xx0_5 = np.zeros(5)

    t_end = 30
    tt = np.linspace(0, t_end, 1000)
  
    
    sol1 = solve_ivp(model1.get_rhs_func(), (0, t_end), xx0_1, t_eval=tt)
    sol2 = solve_ivp(model2.get_rhs_func(), (0, t_end), xx0_2, t_eval=tt)
    sol3 = solve_ivp(model3.get_rhs_func(), (0, t_end), xx0_3, t_eval=tt)
    sol4 = solve_ivp(model4.get_rhs_func(), (0, t_end), xx0_4, t_eval=tt)
    sol5 = solve_ivp(model5.get_rhs_func(), (0, t_end), xx0_5, t_eval=tt)
    
    sim = [sol1, sol2, sol3, sol4, sol5]
    
    save_plot(sim)

    return sim

def save_plot(sim):

    

    # create figure + 2x2 axes array
    fig1, axs = plt.subplots(nrows=2, ncols=2, figsize=(12.8,9.6))
    # print in axes top left
    axs[0, 0].plot(sim[0].t, sim[0].y[0], label = 'PT1')
    axs[0, 0].plot(sim[1].t, sim[1].y[0], label = 'PT2')
    axs[0, 0].set_ylabel('Amplitude') # y-label Nr 1
    axs[0, 0].set_xlabel('Time[s]]') # x-Label für Figure linke Seite
    axs[0, 0].grid()
    axs[0, 0].legend()

    # print in axes top right 
    axs[1, 0].plot(sim[2].t, sim[2].y[0], label = 'PT3')
    axs[1, 0].set_ylabel('Amplitude') # y-label Nr 1
    axs[1, 0].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[1, 0].grid()
    axs[1, 0].legend()

    # print in axes bottom left
    axs[0, 1].plot(sim[3].t, sim[3].y[0], label = 'PT4')
    axs[0, 1].set_ylabel('Amplitude') # y-label Nr 1
    axs[0, 1].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[0, 1].grid()
    axs[0, 1].legend()

    # print in axes bottom right
    axs[1, 1].plot(sim[4].t, sim[4].y[0] , label = 'PT5')
    axs[1, 1].set_ylabel('Amplitude') # y-label Nr 1
    axs[1, 1].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[1, 1].grid()
    axs[1, 1].legend()

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

    target_states = [-2.369685579668774, -1.0626757541773422, -0.6428946899596174, 0.12607608396072245, 0.13375873500072138]

    success = all(abs(simulation_data[i].y[0][-1] - target_states[i]) < 1e-2 for i in np.arange(0, len(simulation_data)))
    
    return ResultContainer(success=success, score=1.0)