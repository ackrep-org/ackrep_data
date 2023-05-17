import numpy as np
import sympy as sp
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import os

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

from ipydex import IPS, activate_ips_on_exception

# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


def simulate():
    """
    simulate model with DAEs, solved by IDA
    return: list with results of the simulation
    """

    # ---------start of edit section--------------------------------------

    model = system_model.Model()

    mod = model.get_mod()
    print("Constraints:\n")
    for i, eq in enumerate(mod.constraints):
        print(eq)
    print("\n")
    print("ODEs:\n")
    for i, eq in enumerate(mod.eqns):
        print(eq)
    print("\n")

    dae_model_func = model.get_dae_model_func(mod)
    # number of configuration coordinates
    ntt = len(mod.tt)

    # initial state values, calculated seperatly
    (yy0, yyd0) = (
        [0.3, 1.74961317, 0.50948621, 0.0, 0.0, 0.0, -0.27535424, 0.5455313],
        [0.0, 0.0, 0.0, 23.53968609, 2.82766884, -14.48960943, -0.0, 0.0],
    )

    t0 = 0

    problem_object = Implicit_Problem(dae_model_func, yy0, yyd0, t0)

    sim = IDA(problem_object)
    sim.verbosity = 0

    tfinal = 11
    ncp = 500

    tt_sol, yy_sol, yyd_sol = sim.simulate(tfinal, ncp)

    ttheta_sol = yy_sol[:, :ntt]
    ttheta_d_sol = yy_sol[:, ntt : ntt * 2]

    simulation_data = [tt_sol, yy_sol, yyd_sol, ttheta_sol, ttheta_d_sol]
    # ---------end of edit section----------------------------------------

    save_plot(simulation_data)

    return simulation_data


def save_plot(simulation_data):
    """
    plot your data and save the plot

    :param simulation_data: simulation_data of system_model
    :return: None
    """
    # ---------start of edit section--------------------------------------
    # plot of your data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9.6))
    plt.sca(ax1)

    ax1.plot(simulation_data[0], simulation_data[3])
    ax1.set_title("angles")

    ax2.plot(simulation_data[0], simulation_data[4])
    ax2.set_title("angular velocities")

    # ---------end of edit section----------------------------------------

    plt.tight_layout()

    save_plot_in_dir()


def evaluate_simulation(simulation_data):
    """
    assert that the simulation results are as expected

    :param simulation_data: simulation_data of system_model
    :return: ResultContainer
    """
    # ---------start of edit section--------------------------------------
    # fill in final states of simulation to check your model
    # simulation_data.y[i][-1]
    expected_final_state = [11, 0.07655823245005482, -11.441290258055378, -4.092516225352934, 1.6614692473252495]

    # [6, 0.9774883908572787, 0.9398128300737418, -4.229675196768807, -0.15115736529169507]

    # ---------end of edit section----------------------------------------

    rc = ResultContainer(score=1.0)
    simulated_final_state = [
        simulation_data[0][-1],
        simulation_data[1][-1][-1],
        simulation_data[2][-1][-1],
        simulation_data[3][-1][-1],
        simulation_data[4][-1][-1],
    ]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc
