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
        print(f"dot_x{i} =", eq)

    rhs = model.get_rhs_func()

    ## TODO: ---- paste simulation data of from ..._test.py here----------
    # Initial State values  
    xx0 = ...

    t_end = ...
    tt = ...
    sim = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)
    # if inputfunction exists:
    uu = ...
    sim.uu = uu


    # --------------------------------------------------------------------
    
    save_plot(sim)

    return sim

def save_plot(simulation_data):
    ## TODO: ---- paste plotting data of ..._test.py here ----------------
    # access to data via:
    simulation_data.t
    simulation_data.y
    simulation_data.uu
    plt.plot(...)
    # etc.


    # --------------------------------------------------------------------

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
    ## TODO: --- calculate final states of simulation --------------------
    # run ..._test.py and print final states y[i][-1]
    # copy paste those values to target_states
    target_states = [...]
    
    # --------------------------------------------------------------------

    success = all(abs(simulation_data.y[i][-1] - target_states[i]) < 1e-2 for i in np.arange(0, len(simulation_data.y)))
    
    return ResultContainer(success=success, score=1.0)