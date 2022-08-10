import numpy as np
import pyinduct as pi
import pyqtgraph as pg
import system_model
from scipy.integrate import solve_ivp

from ackrep_core import ResultContainer
from ackrep_core.system_model_management import save_plot_in_dir
import matplotlib.pyplot as plt
import matplotlib
import os

from ipydex import IPS

# link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


def simulate():
    """
    simulate the system model with scipy.integrate.solve_ivp

    :return: result of solve_ivp, might contains input function
    """

    model = system_model.Model()

    print(">>> derive initial conditions")
    q0 = pi.core.project_on_bases(model.initial_states, model.canonical_equations)

    print(">>> perform time step integration")
    sim_domain, q = pi.simulate_state_space(model.state_space_form, q0, model.temp_domain, settings=None)

    print(">>> perform postprocessing")
    eval_data = pi.get_sim_results(sim_domain, model.spatial_domains, q, model.state_space_form, derivative_orders=model.derivative_orders)

    evald_x = pi.evaluate_approximation(model.func_label, q[:,:model.n], sim_domain, model.spat_domain, name="x(z,t)")

    pi.tear_down(labels=(model.func_label,))


    sim = ResultContainer()
    sim.u = model.u.get_results(eval_data[0].input_data[0]).flatten()
    sim.eval_data = eval_data
    sim.evald_x = evald_x

    save_plot(sim)

    return sim


def save_plot(simulation_data):
    """
    plot your data and save the plot
    access to data via: simulation_data.t   array of time values
                        simulation_data.y   array of data components
                        simulation_data.uu  array of input values

    :param simulation_data: simulation_data of system_model
    :return: None
    """
    # Note: plotting in pyinduct is usually done with pyqtgraph which causes issues during CI.
    # This is why the plotting part doesnt look as clean.
    # Pyinduct has its own plotting methods, feel free to use them in your own implementation.
    matplotlib.use("Agg")

    # input visualization
    win0 = plt.plot(np.array(simulation_data.eval_data[0].input_data[0]).flatten(),
                simulation_data.u)
    plt.title("Input Trajectory at $z=2\pi$")
    plt.xlabel("Time $t$")
    plt.ylabel("$u(t)$")
    plt.tight_layout()
    save_plot_in_dir("plot_1.png")

    win1 = pi.surface_plot(simulation_data.evald_x, zlabel="$x(z,t)$")
    plt.title("Propagation of the Wave in Time and Space")
    plt.ylabel("Time $t$")
    plt.xlabel("Space $z$")
    plt.tight_layout()
    save_plot_in_dir("plot_2.png")
    
    # vis.save_2d_pg_plot(win0, 'transport_system')
    # win1 = pi.PgAnimatedPlot(simulation_data.eval_data,
    #                             title=simulation_data.eval_data[0].name,
    #                             save_pics=False,
    #                             labels=dict(left='x(z,t)', bottom='z'))
    # pi.show()



def evaluate_simulation(simulation_data):
    """
    assert that the simulation results are as expected

    :param simulation_data: simulation_data of system_model
    :return:
    """
    expected_final_state = np.array([
        -0.00499123,  0.00998247,  0.02509866,  0.04167615,  0.05997688,
        0.08077127,  0.1049973 ,  0.13320243,  0.16511553,  0.20080206,
        0.24051338,  0.28455143,  0.33169568,  0.38166332,  0.4345756 ,
        0.48950293,  0.54560654,  0.60265054,  0.66059809,  0.7191791 ,
        0.77863313,  0.83889479,  0.90044527,  0.96326604,  1.02625365,
        1.08924296,  1.15218586,  1.21474135,  1.27650887,  1.33697007,
        1.39539469,  1.4512837 ,  1.50533233,  1.55766056,  1.60842412,
        1.65739483,  1.70426427,  1.74852285,  1.78923102,  1.82615575,
        1.8593328 ,  1.88908859,  1.91531896,  1.93833024,  1.95859756,
        1.97552569,  1.98838818,  1.99653955,  2.00014191,  1.99954084,
        1.99504644,  1.98740683,  1.97630684,  1.96149697,  1.94305579,
        1.9208185 ,  1.89439842,  1.863304  ,  1.8285013 ,  1.79124462,
        1.7517036 ,  1.70961109,  1.66492066,  1.61726217,  1.56689181,
        1.51390145,  1.45842196,  1.40085832,  1.34170291,  1.28171429,
        1.22104533,  1.15970548,  1.0978266 ,  1.03579091,  0.97373471,
        0.91219186,  0.85053241,  0.78807755,  0.72575174,  0.66438022,
        0.60454764,  0.54669319,  0.4912745 ,  0.43833473,  0.38835524,
        0.34127898,  0.29700688,  0.25579216,  0.21756052,  0.18142819,
        0.14731514,  0.11535209,  0.085886  ,  0.05901843,  0.03463059,
        0.01302813, -0.00517915, -0.01954857, -0.03013463, -0.03678862,
       -0.03905366
    ])

    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.eval_data[0].output_data[-1]
    rc.final_state_errors = [
        simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))
    ]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)

    return rc