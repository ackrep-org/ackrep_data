# This file was autogenerated from the template: simulation.py.template (2022-10-10 15:54:03).

import numpy as np
import system_model
from scipy.integrate import solve_ivp, odeint

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os
from ipydex import Container

# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


def simulate():
    """
    simulate the system model with scipy.integrate.solve_ivp

    :return: result of solve_ivp, might contains input function
    """

    model = system_model.Model()

    rhs_xx_pp_symb = model.get_rhs_symbolic()
    rhs = model.get_rhs_func()

    # initial state values
    xx0 = np.ones(model.sys_dim)

    t_end = 10
    tt = np.linspace(0, t_end, 1000)

    simulation_data = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)

    # using odeint for models with large state vectors
    # res = odeint(rhs, y0=xx0, t=tt, tfirst=True)
    # simulation_data = Container()
    # simulation_data.y = res.transpose()
    # simulation_data.t = tt

    # postprocessing: calc output
    ny = 2
    C = model.get_parameter_value("C")
    D21 = model.get_parameter_value("D21")
    output = np.zeros((ny, len(tt)))
    for i in range(len(tt)):
        output[:,i] = np.matmul(C, simulation_data.y[:,i]) # + np.matmul(D21, w)
    simulation_data.output = output

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

    for i in range(simulation_data.output.shape[0]):
        plt.plot(simulation_data.t, simulation_data.output[i], label=f"$y_{i}$")

    plt.legend()
    plt.tight_layout()

    save_plot_in_dir()


def evaluate_simulation(simulation_data):
    """
    assert that the simulation results are as expected

    :param simulation_data: simulation_data of system_model
    :return:
    """
    expected_final_state = np.array([5.68533159e-01, 4.68985195e-01, 4.59567784e-05])

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc