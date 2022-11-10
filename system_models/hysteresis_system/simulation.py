import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os
from pyblocksim import *

# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


def simulate():
    """
    simulate the system model with blocks

    :return: time, hysteresis output, input signal
    """

    model = system_model.Model()

    #input
    uu = model.input_ramps

    Result_block, hyst_in = model.get_blockfnc()
    
    tt, states = blocksimulation(25, {hyst_in: uu}) # simulate
    tt = tt.flatten()

    bo = compute_block_ouptputs(states)

    input_signal = [uu(t) for t in tt]

    simulation_data = [tt, bo[Result_block], input_signal]


    save_plot(simulation_data)

    return simulation_data


def save_plot(simulation_data):
    """
    plot your data and save the plot

    :param simulation_data: simulation_data of system_model
    :return: None
    """
    # ---------start of edit section--------------------------------------
    # plot of your data
    fig1, axs = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 9.6))

    # print in axes top left

    axs[0].plot(simulation_data[0], simulation_data[2], label='input')
    axs[0].plot(simulation_data[0], simulation_data[1], label='hyst. output')
    axs[0].set_xlabel("Time [s]")  # x-label
    axs[0].grid(1)
    axs[0].legend()

    # print in axes top right
    axs[1].plot(simulation_data[2], simulation_data[1])
    axs[1].set_xlabel("Input signal")  # y-label
    axs[1].set_ylabel("Hysteresis-output")  # x-Label
    axs[1].grid(1)

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
    expected_final_state = [25, 2, 0]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = [simulation_data[0][-1], simulation_data[1][-1], simulation_data[2][-1]]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc


