

import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os

import time
from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()


def simulate():
    model = system_model.Model()

    rhs_xx_pp_symb = model.get_rhs_symbolic()
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i+1} =", eq)

    rhs = model.get_rhs_func()

    # --------------------------------------------------------------------
    
    xx0 = [0, 0, 0+0j, 0+0j, 0+0j, 0, 0+0j, 0]
    t_end = 3
    tt = np.linspace(0, t_end, 3000)
    # use model class rhs
    sol = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

    i = 0
    n_rows = len(sol.t)
    n_cols = 4
    uu = np.zeros((n_rows, n_cols))

    while i < len(sol.t):
        uu[i, :] = model.uu_func(sol.t[i], sol.y[:, i])
        i = i+1

    sol.uu = uu

    # --------------------------------------------------------------------
    
    save_plot(sol)

    return sol  

def save_plot(sol):

    # --------------------------------------------------------------------

    # create figure + 2x2 axes array
    fig1, axs = plt.subplots(nrows=2, ncols=2, figsize=(12.8,9.6))

    # print in axes top left
    axs[0, 0].plot(sol.t, np.real(sol.y[1] ), label = 'Re' )
    axs[0, 0].set_ylabel('ed0') # y-label 
    axs[0, 0].set_xlabel('Time[s]') # x-Label 
    axs[0, 0].grid()
    axs[0, 0].legend()

    # print in axes top right 
    axs[1, 0].plot(sol.t, np.real(sol.y[2] ), label = 'Re')
    axs[1, 0].plot(sol.t, np.imag(sol.y[2] ), label = 'Im')
    axs[1, 0].set_ylabel('es') # y-label 
    axs[1, 0].set_xlabel('Time[s]') # x-Label 
    axs[1, 0].grid()
    axs[1, 0].legend()

    # print in axes bottom left
    axs[0, 1].plot(sol.t, np.real(sol.y[3] ), label = 'Re')
    axs[0, 1].plot(sol.t, np.imag(sol.y[3] ), label = 'Im')
    axs[0, 1].set_ylabel('ed') # y-label 
    axs[0, 1].set_xlabel('Time[s]') # x-Label 
    axs[0, 1].grid()
    axs[0, 1].legend()

    # print in axes bottom right
    axs[1, 1].plot(sol.t, sol.uu[:, 0] , label = 'vy')
    axs[1, 1].plot(sol.t, sol.uu[:, 1] , label = 'vy0')
    axs[1, 1].set_ylabel('') # y-label 
    axs[1, 1].set_xlabel('Time[s]') # x-Label 
    axs[1, 1].grid()
    axs[1, 1].legend()

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
    #--------------------------------------------------------------------
    # fill in final states of simulation to check your model
    # simulation_data.y[i][-1]
    expected_final_state = [(55.43841456212572+0j), (20.413747733428224+0j), (35.79509623855708-75.96575244375583j), (-72.91879818170683+485.1484950343515j), 0j, (17.557881788491308+0j), (9.991045345136369-3.6215387749437133j), (94.24777960769383+0j)]
    
    # --------------------------------------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)
    
    return rc