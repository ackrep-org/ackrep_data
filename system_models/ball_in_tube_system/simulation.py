import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os

# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


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
    xx0 = [456, 0, 0]

    t_end = 10
    tt = np.linspace(0, t_end, 1000)
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

    # create figure + 2x2 axes array
    fig1, axs = plt.subplots(nrows=3, ncols=1, figsize=(12.8, 9.6))

    axs[0].plot(simulation_data.t, simulation_data.y[1])
    axs[0].set_ylabel("Height of the ball [m]")  # y-label
    axs[0].grid()

    axs[1].plot(simulation_data.t, simulation_data.y[2])
    axs[1].set_ylabel("Velocity of the ball [m/s]")  # y-label
    axs[1].grid()

    axs[2].plot(simulation_data.t, simulation_data.y[0])
    axs[2].set_ylabel("Rotation speed [U/min]")  # y-label
    axs[2].set_xlabel("Time [s]")  # x-Label
    axs[2].grid()

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
    expected_final_state = [1965.06846972, 2.1096, 0.2131639]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
