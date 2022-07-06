

import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
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
    xx0 = np.zeros(6)

    t_end = 20
    tt = np.linspace(0, t_end, 10000)
    sim = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt, max_step=0.01)

    # if inputfunction exists:
    uu = model.uu_func(sim.t, sim.y)
    g = model.pp_symb[0]
    m = model.pp_symb[2]
    uu = np.array(uu)/(model.pp_dict[g]*model.pp_dict[m])   
    sim.uu = uu

    # --------------------------------------------------------------------
    
    save_plot(sim)

    return sim

def save_plot(simulation_data):
    
    # create figure + 2x2 axes array
    fig1, axs = plt.subplots(nrows=3, ncols=1, figsize=(12.8,12))

    # print in axes top left 
    axs[0].plot(simulation_data.t, np.real(simulation_data.y[0] ), label = 'x-Position' )
    axs[0].plot(simulation_data.t, np.real(simulation_data.y[2] ), label = 'y-Position' )
    axs[0].plot(simulation_data.t, np.real(simulation_data.y[4]*180/np.pi ), label = 'angle' )
    axs[0].set_title('Position')
    axs[0].set_ylabel('Position[m]') # y-label Nr 1
    axs[0].set_xlabel('Time[s]') 
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(simulation_data.t, simulation_data.y[1], label = 'v_x')
    axs[1].plot(simulation_data.t, simulation_data.y[3], label = 'v_y')
    axs[1].plot(simulation_data.t, simulation_data.y[5]*180/np.pi , label = 'angular velocity')
    axs[1].set_title('Velocities')
    axs[1].set_ylabel('Velocity[m/s]')
    axs[1].set_xlabel('Time[s]')
    axs[1].grid()
    axs[1].legend()

    # print in axes bottom left
    axs[2].plot(simulation_data.t, simulation_data.uu[0] , label = 'Force left')
    axs[2].plot(simulation_data.t, simulation_data.uu[1] , label = 'Force right')
    axs[2].set_title('Normalized Input Forces')
    axs[2].set_ylabel('Forces normalized to F_g') # y-label Nr 1
    axs[2].set_xlabel('Time[s]') 
    axs[2].grid()
    axs[2].legend()

    # adjust subplot positioning and show the figure
    fig1.subplots_adjust(hspace=0.5)
    #fig1.show()

    # --------------------------------------------------------------------

    plt.tight_layout()

    save_plot_in_dir(os.path.dirname(__file__), plt)

def evaluate_simulation(simulation_data):
    """
    
    :param simulation_data: simulation_data of system_model
    :return:
    """
    # fill in the final states y[i][-1] to check your model
    expected_final_state = [-44.216119976296774, -3.680370979746213, 45.469521639337344, -43.275661598545256, -0.00037156407418776797, -0.00033632680535548506]
    
    # --------------------------------------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)
    
    return rc