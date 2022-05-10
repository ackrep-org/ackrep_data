

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

    # --------------------------------------------------------------------
    
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

    # --------------------------------------------------------------------

    # plot of your data
    # access to data via:
    simulation_data.t
    simulation_data.y
    simulation_data.uu
    plt.plot(...)


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
    #--------------------------------------------------------------------
    # fill in final states of simulation to check your model
    # simulation_data.y[i][-1]
    target_states = [...]
    
    # --------------------------------------------------------------------

    rc = ResultContainer(score=1.0)
    rc.target_state_errors = [simulation_data.y[i][-1] - target_states[i] for i in np.arange(0, len(simulation_data.y))]
    rc.success = all(abs(np.array(rc.target_state_errors)) < 1e-2)
    
    return rc