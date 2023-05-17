from cProfile import label
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
    xx0 = np.zeros(10)

    t_end = 3
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
    fig1, axs = plt.subplots(nrows=10, ncols=1, figsize=(12.8, 9))

    # print in axes top left
    axs[0].plot(simulation_data.t, simulation_data.y[0])
    axs[0].set_ylabel("p1")  # y-label
    axs[0].grid()

    axs[1].plot(simulation_data.t, simulation_data.y[1])
    axs[1].set_ylabel("p2")  # y-label
    axs[1].grid()

    axs[2].plot(simulation_data.t, simulation_data.y[2])
    axs[2].set_ylabel("p3")  # y-label
    axs[2].grid()

    axs[3].plot(simulation_data.t, simulation_data.y[3])
    axs[3].set_ylabel("q1")  # y-label
    axs[3].grid()

    axs[4].plot(simulation_data.t, simulation_data.y[4])
    axs[4].set_ylabel("q2")  # y-label
    axs[4].grid()

    axs[5].plot(simulation_data.t, simulation_data.y[5])
    axs[5].set_ylabel(r"$dot{p}_1$")  # y-label
    axs[5].grid()

    axs[6].plot(simulation_data.t, simulation_data.y[6])
    axs[6].set_ylabel(r"$dot{p}_2$")  # y-label
    axs[6].grid()

    axs[7].plot(simulation_data.t, simulation_data.y[7])
    axs[7].set_ylabel(r"$dot{p}_3$")  # y-label
    axs[7].grid()

    axs[8].plot(simulation_data.t, simulation_data.y[8])
    axs[8].set_ylabel(r"$dot{q}_1$")  # y-label
    axs[8].grid()

    axs[9].plot(simulation_data.t, simulation_data.y[9])
    axs[9].set_ylabel(r"$dot{q}_2$")  # y-label
    axs[9].grid()
    # ---------end of edit section----------------------------------------

    plt.tight_layout()

    # plt.show()

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
    expected_final_state = [0.0, -44.14499999999994, 0.0, 0.0, 0.0, 0.0, -29.429999999999964, 0.0, 0.0, 0.0]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
