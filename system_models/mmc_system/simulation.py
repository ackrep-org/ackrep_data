

import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
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
    start = time.time()
    
    xx0 = [0, 0, 0+0j, 0+0j, 0+0j, 0, 0+0j, 0]
    t_end = 3
    tt = np.linspace(0, t_end, 3000)
    # use model class rhs
    sol = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

    print("Dauer[s]: ", time.time() - start)
  
    start = time.time()
    i = 0
    n_rows = len(sol.t)
    n_cols = 4
    uu = np.zeros((n_rows, n_cols))

    while i < len(sol.t):
        #IPS()
        uu[i, :] = model.uu_func(sol.t[i], sol.y[:, i])
        i = i+1


    sol.uu = uu

    print("Dauer[s]: ", time.time() - start)

    # --------------------------------------------------------------------
    
    save_plot(sol)

    return sol  

def save_plot(sol):

    # --------------------------------------------------------------------

    # create figure + 2x2 axes array
    fig1, axs = plt.subplots(nrows=2, ncols=2, figsize=(12.8,9.6))

    # print in axes top left
    axs[0, 0].plot(sol.t, np.real(sol.y[1] ), label = 'Re' )
    axs[0, 0].set_ylabel('ed0') # y-label Nr 1
    axs[0, 0].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[0, 0].grid()
    axs[0, 0].legend()

    # print in axes top right 
    axs[1, 0].plot(sol.t, np.real(sol.y[2] ), label = 'Re')
    axs[1, 0].plot(sol.t, np.imag(sol.y[2] ), label = 'Im')
    axs[1, 0].set_ylabel('es') # y-label Nr 1
    axs[1, 0].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[1, 0].grid()
    axs[1, 0].legend()

    # print in axes bottom left
    axs[0, 1].plot(sol.t, np.real(sol.y[3] ), label = 'Re')
    axs[0, 1].plot(sol.t, np.imag(sol.y[3] ), label = 'Im')
    axs[0, 1].set_ylabel('ed') # y-label Nr 1
    axs[0, 1].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[0, 1].grid()
    axs[0, 1].legend()

    # print in axes bottom right
    axs[1, 1].plot(sol.t, sol.uu[:, 0] , label = 'vy')
    axs[1, 1].plot(sol.t, sol.uu[:, 1] , label = 'vy0')
    axs[1, 1].set_ylabel('') # y-label Nr 1
    axs[1, 1].set_xlabel('Time[s]') # x-Label für Figure linke Seite
    axs[1, 1].grid()
    axs[1, 1].legend()

    # adjust subplot positioning and show the figure
    #fig1.suptitle('Simulationen des geschlossenen Kreises, Sprunganregung', fontsize=16)
    fig1.subplots_adjust(hspace=0.5)
    fig1.show()


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
    expected_final_state = [56.00010840992848+0j, 9.241054675268499+0j, 9.463364142654473-18.73632876714279j, -47.55520379428192+131.90680874695636j, 0j, 16.504924039370916+0j, 9.99999999999999+1.7084490323538374e-17j, 125.66370614359165+0j]
    
    # --------------------------------------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)
    
    return rc