

import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
import matplotlib.pyplot as plt
import os

#link to documentation with examples: 
#


def simulate():
    """
    simulate the system model with scipy.integrate.solve_ivp
         
    :return: result of solve_ivp, might contains input function
    """ 

    model = system_model.Model()

    rhs_xx_pp_symb = model.get_rhs_symbolic()
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i+1} =", eq)

    rhs = model.get_rhs_func()

    # ---------start of edit section--------------------------------------
    # initial state values  
    xx0 = [0, 0, 0, 0]

    t_end = 30
    tt = np.linspace(0, t_end, 10000)
    simulation_data = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

    # define inputfunction
    #uu = ...        #uu = model.uu_func(simulation_data.t, ...)
    #simulation_data.uu = uu
    # ---------end of edit section----------------------------------------
    
    save_plot(simulation_data)

    return simulation_data  

def save_plot(simulation_data):
    """
    plot your data and save the plot
    access to data via: simulation_data.t   array of time values
                        simulation_data.y   array of data components 
                        simulation_data.uu  array of input values 

    :param simulation_data: simulation_data of system_model     
    :return: None
    """ 
    # ---------start of edit section--------------------------------------
    

    # print in axes top left
    plt.plot(simulation_data.t, simulation_data.y[0]+1.8*np.sin(simulation_data.y[1]), label = 'x position of the last')
    plt.plot(simulation_data.t, simulation_data.y[0], label = 'x postion of the wagon')
    plt.ylabel('x[m]') # y-label
    plt.xlabel('Time[s]') # x-label
    plt.grid()
    plt.legend()


    plt.tight_layout()

    # ---------end of edit section----------------------------------------

    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(__file__), '_system_model_data')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'plot.png'), dpi=96 * 2)

def evaluate_simulation(simulation_data):
    """
    assert that the simulation results are as expected

    :param simulation_data: simulation_data of system_model
    :return:
    """
    # ---------start of edit section--------------------------------------
    # fill in final states of simulation to check your model
    # simulation_data.y[i][-1]
    expected_final_state = [10, 10, 2, 2]
    
    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.target_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state)
    
    return rc