import numpy as np
import system_model
from scipy.integrate import solve_ivp
from pyblocksim import *

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os

# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


def simulate():
    """
    simulate the system model 

    :return: simulation data
    """

    model = system_model.Model()

    # rhs_xx_pp_symb = model.get_rhs_symbolic()
    # print("Computational Equations:\n")
    # for i, eq in enumerate(rhs_xx_pp_symb):
    #     print(f"dot_x{i+1} =", eq)

    
    # ---------start of edit section--------------------------------------

    SUM1, SUM2, u3 = model.get_Blockfnc()

    thestep = stepfnc(1.0, 1)

    t, states = blocksimulation(40, (u3, thestep), dt=.05)

    bo = compute_block_ouptputs(states)

    simulation_data = [t, bo[SUM1], bo[SUM2]]

    # ---------end of edit section----------------------------------------

    save_plot(simulation_data)

    return simulation_data


def save_plot(simulation_data):
    """
    plot data and save the plot

    :param simulation_data: simulation_data of system_model
    :return: None
    """
    # ---------start of edit section--------------------------------------
    # plot of your data
    plt.plot(simulation_data[0], simulation_data[1], label='Filling level')
    plt.plot(simulation_data[0], simulation_data[2], label='Temperature')
    plt.xlabel('Time[s]')
    plt.legend()
    plt.grid()
    # ---------end of edit section----------------------------------------

    plt.tight_layout()

    save_plot_in_dir()


def evaluate_simulation(simulation_data):
    """
    assert that the simulation results are as expected

    :param simulation_data: simulation_data of system_model
    :return:
    """
    # ---------start of edit section--------------------------------------
    # fill in final states of simulation to check your model
    # simulation_data.y[i][-1]
    expected_final_state = [40.05, 0.99676226, -2.16627258e-03]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = [simulation_data[0][-1], simulation_data[1][-1], simulation_data[2][-1]]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
