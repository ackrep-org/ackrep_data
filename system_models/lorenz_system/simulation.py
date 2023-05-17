# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:06:37 2021

@author: Rocky
"""

import numpy as np
import system_model as lac
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os


def simulate():
    # Defining Input functions
    lorenz_att = lac.Model()

    rhs_xx_pp_symb = lorenz_att.get_rhs_symbolic()
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i+1} =", eq)

    latt_rhs = lorenz_att.get_rhs_func()

    # Initial State values
    xx0 = [0.1, 0.1, 0.1]

    t_end = 30
    tt = np.linspace(0, t_end, 10000)  # vector of times for simulation
    sol = solve_ivp(latt_rhs, (0, t_end), xx0, t_eval=tt)

    save_plot(sol)

    return sol


def save_plot(simulation_data):
    fig = plt.figure(figsize=(10, 4.5))
    ax = plt.axes(projection="3d")
    ax.plot(simulation_data.y[0], simulation_data.y[1], simulation_data.y[2])
    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
    plt.title("Full Simulation")

    plt.tight_layout()

    save_plot_in_dir()


def evaluate_simulation(simulation_data):
    """

    :param simulation_data: simulation_data of system_model
    :return:
    """

    expected_final_state = [3.30299301, 4.43659481, 17.91529214]
    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
