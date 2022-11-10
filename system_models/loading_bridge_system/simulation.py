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

    print("The input function only consists of the positive half wave of a sinus function. Otherwise it is zero.\n")

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

    u =[]
    for i in range(len(simulation_data.t)):
        u.append(model.uu_func(simulation_data.t[i], xx0)[0])
    simulation_data.uu = u

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
    fig1, axs = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 9))

    # print in axes top left
    axs[0].plot(simulation_data.t, simulation_data.y[0] + 1.8 * np.sin(simulation_data.y[1]), label="x position of the last")
    axs[0].plot(simulation_data.t, simulation_data.y[0], label="x postion of the cart")
    axs[0].set_ylabel("x [m]")  # y-label
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(simulation_data.t, simulation_data.uu)
    axs[1].set_ylabel("u [N]")  # y-label
    axs[1].set_xlabel("Time [s]") # x-label
    axs[1].grid()

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
    expected_final_state = [19.113936156609768, -0.0023747095479253674, 0.04470903934877705, -0.21878088945846508]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
