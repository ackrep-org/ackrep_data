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

    # initial state values
    xx0 = [0, 0, 0, 0]

    t_end = 8
    tt = np.linspace(0, t_end, 10000)
    simulation_data = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

    u =[]
    for i in range(len(simulation_data.t)):
        u.append(model.uu_func(simulation_data.t[i], xx0)[0])
    simulation_data.uu = u

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
    fig1, axs = plt.subplots(nrows=5, ncols=1, figsize=(12.8, 9))

    # print in axes top left
    axs[0].plot(simulation_data.t, simulation_data.y[0])
    axs[0].set_ylabel(r"$x_1 \, [rad]$")  # y-label
    axs[0].grid()

    axs[1].plot(simulation_data.t, simulation_data.y[1])
    axs[1].set_ylabel(r"$x_2 \, [rad]$")  # y-label
    axs[1].grid()

    axs[2].plot(simulation_data.t, simulation_data.y[2])
    axs[2].set_ylabel(r"$x_3 \, [rad/s]$")  # y-label
    axs[2].grid()

    axs[3].plot(simulation_data.t, simulation_data.y[3])
    axs[3].set_ylabel(r"$x_4 \, [rad/s]$")  # y-label
    axs[3].grid()

    axs[4].plot(simulation_data.t, simulation_data.uu)
    axs[4].set_ylabel(r"$u_1 \, [Nm]$")  # y-label
    axs[4].set_xlabel("Time [s]")  # x-label
    axs[4].grid()

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
    expected_final_state = [149.755395984016, -1.15553895270285, 35.92443402010459, 6.114497336949881]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
