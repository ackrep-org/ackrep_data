# This file was autogenerated from the template: simulation.py.template (2022-10-10 15:53:27).

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
    ny = 5
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
    expected_final_state = np.array([ 3.50417944e-02,  4.87112538e-02,  1.26796600e-03,  2.05084191e-04,
       -4.81577324e-05,  1.31301895e-03,  8.73277217e-04,  5.48833050e-03,
        2.26016575e-03,  7.68139877e-03,  1.87863751e-03, -2.90543492e-04,
       -3.99464498e-05, -1.72153920e-03,  9.29115393e-04, -9.38333054e-04,
        4.74309697e-44, -4.74309697e-43,  8.55715385e-16,  1.42500013e-14,
       -1.26565762e-13,  2.29108245e-11, -5.63730621e-11,  1.38708327e-10,
        3.57376470e-02,  5.03412618e-02,  1.31039117e-03, -4.97668473e-05,
        1.26521807e+00, -2.30537148e-01])

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc