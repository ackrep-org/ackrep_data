import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os

# link to documentation with examples:
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
    xx0 = [0.02, 0.02, 0, 0]                     #0.02, 0.052, 0, 0]

    t_end = 2
    tt = np.linspace(0, t_end, 10000)
    simulation_data = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

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
    # plot of your data
    plt.plot(simulation_data.t, simulation_data.y[0], label="position of the iron ball in x-direction")
    plt.plot(simulation_data.t, simulation_data.y[1], label="position of the brass ball in x-direction")
    plt.plot(simulation_data.t, simulation_data.y[2], label="velocity of the iron ball in x-direction")
    plt.plot(simulation_data.t, simulation_data.y[3], label="velocity of the brass ball in x-direction")
    plt.xlabel("Time [s]")  # x-label
    plt.grid()
    plt.legend()

    # ---------end of edit section----------------------------------------

    plt.tight_layout()

    plt.show()

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
    expected_final_state = [14.623557063331738, 14.696058924396356, 16.881612845239694, 16.981880259096393]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
