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
    xx0 = [2,3,4]
    t_end = 300
    tt = np.linspace(0,t_end,6000)
    sim = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)
    
    save_plot(sim)

    return sim

def save_plot(simulation_data):
    
    y = simulation_data.y.tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0],y[1],y[2],label='Phasenportrait',lw=1,c='k')
    #pyplot.title('Zustandsverläufe')
    ax.set_xlabel('x',fontsize= 15)
    ax.set_ylabel('y',fontsize= 15)
    ax.set_zlabel('z',fontsize= 15)
    ax.legend()
    ax.grid()

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

    target_states = [-6.186104528534191, -0.41409773532206806, -0.16828546564818292]

    success = all(abs(simulation_data.y[i][-1] - target_states[i]) < 1e-2 for i in np.arange(0, len(simulation_data.y)))
    
    return ResultContainer(success=success, score=1.0)