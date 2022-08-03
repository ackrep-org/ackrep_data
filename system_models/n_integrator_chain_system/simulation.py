# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:06:37 2021

@author: Rocky
"""

import numpy as np
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os


def simulate():

    order = 5
    model_two = system_model.Model(order)

    u = model_two.uu_func

    """RHS-Function of the brockett_model
        Args:
            t (float, positive): time
            x (ndarray): state vector
        Returns:
            dx/dt (ndarray): time derivative of state vector at time t
    """

    def chain_integrator_model(t, xx_nv):

        dxx_dt_nv = xx_nv * 1
        dxx_dt_nv[:-1] = xx_nv[1:]
        dxx_dt_nv[-1] = u(t, xx_nv)

        return dxx_dt_nv

    xx0 = [0, 0, 0, 0, 0]

    model = system_model.Model(x_dim=len(xx0))

    rhs_xx_pp_symb = model.get_rhs_symbolic()
    print("Computational Equations:\n")
    for i, eq in enumerate(rhs_xx_pp_symb):
        print(f"dot_x{i+1} =", eq)

    rhs = model.get_rhs_func()

    t_end = 30
    tt = times = np.linspace(0, t_end, 1000)
    sim = solve_ivp(chain_integrator_model, (0, t_end), xx0, t_eval=tt)

    save_plot(sim)

    return sim


def save_plot(sim):

    plt.plot(sim.t, sim.y[0], label="x1", lw=1)
    plt.plot(sim.t, sim.y[1], label="x2", lw=1)
    plt.plot(sim.t, sim.y[2], label="x3", lw=1)
    plt.plot(sim.t, sim.y[3], label="x4", lw=1)
    plt.plot(sim.t, sim.y[4], label="x5", lw=1)

    plt.title("State progress")
    plt.xlabel("Time[s]", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.legend()
    plt.grid()

    plt.tight_layout()

    save_plot_in_dir()


def evaluate_simulation(simulation_data):
    """

    :param simulation_data: simulation_data of system_model
    :return:
    """

    expected_final_state = [
        32337.334058261687,
        3092.752509636077,
        144.88833045860406,
        -1.4410336827947148,
        -0.0673030971647946,
    ]

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.y[:, -1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
