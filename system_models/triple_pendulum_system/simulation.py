import numpy as np
import system_model
from scipy.integrate import odeint

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

    mod = model.get_symbolic_model()
    print("Computational Equations:\n")
    for i, eq in enumerate(mod.eqns):
        print(f"dot_x{i+1} =", eq)

    rhs = model.get_rhs_odeint_fnc()

    # initial state values
    xx0 = np.array([2, 2, 2, 0, 0, 0, 0, 0])

    t_end = 15
    tt = np.linspace(0, t_end, 10000)
    simulation_data = odeint(rhs, xx0, tt) 

    save_plot(simulation_data, tt)

    return simulation_data


def save_plot(simulation_data, tt):
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
    for i in range(len(simulation_data[0,:])):
        plt.plot(tt, simulation_data[:,i], label='$x_{}$'.format(i+1))
    plt.grid()
    plt.legend()
    plt.xlabel('Time [s]')

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
    expected_final_state = [3.3544161195899775, 3.6128436581303522, 4.184663788192004, 0.0, -2.0778538809498333, -5.473145553078993, -2.3347177255025864, 0.0]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = []
    for i in range(simulation_data.shape[1]):
        simulated_final_state.append(simulation_data[:, i][-1])

    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
