

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

#link to documentation with examples: https://ackrep-doc.readthedocs.io/en/latest/devdoc/contributing_data.html


def simulate():
    """
    simulate the system model with scipy.integrate.solve_ivp
         
    :return: result of solve_ivp, might contains input function
    """ 

    model = system_model.Model()

    print(">>> derive initial conditions")
    q0 = pi.core.project_on_bases(model.initial_states, model.canonical_equations)

    print(">>> perform time step integration")
    sim_domain, q = pi.simulate_state_space(model.state_space_form, q0, model.temp_domain,
                                            settings=None)

    print(">>> perform postprocessing")
    eval_data = pi.get_sim_results(sim_domain, model.spatial_domains, q, model.state_space_form,
                                derivative_orders=model.derivative_orders)

    evald_x = pi.evaluate_approximation(
        model.func_label,
        q,
        sim_domain, 
        model.spat_domain,
        name="x(z,t)")

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
    # todo: implement functionality to support multiple plots --> core
    # imput data
    # win0 = plt.plot(np.array(simulation_data.eval_data[0].input_data[0]).flatten(),
    #             simulation_data.u)
    # plt.title("Input function at z=0")
    # plt.xlabel("t [s]")
    # plt.ylabel("u(t)")
    # save_plot_in_dir("input.png")

    matplotlib.use('Agg')
    win1 = pi.surface_plot(simulation_data.evald_x, zlabel=simulation_data.evald_x.name)
    # save_plot_in_dir("sim.png")
    save_plot_in_dir()

    # Animation, try it yourself!
    # win1 = pi.PgAnimatedPlot(simulation_data.eval_data,
    #                         title=simulation_data.eval_data[0].name,
    #                         save_pics=False,
    #                         labels=dict(left='x(z,t)', bottom='z'))  
    # pi.show()   


def evaluate_simulation(simulation_data):
    """
    assert that the simulation results are as expected

    :param simulation_data: simulation_data of system_model
    :return:
    """
    expected_final_state = np.array([15.9121583 , 16.09994978, 15.96335188, 13.08588575, 10.13711019,
        9.6499634 , 10.67303024,  9.56644162,  9.6876343 , 10.81433698,
        9.47145433,  9.78543658, 10.2484261 , 10.46145942, 10.07941574,
        8.39027042, 10.62520391, 15.32583395, 18.08964   , 20.24542305,
       20.23232229, 19.50687173, 20.68567787, 19.68595808, 19.74016142,
       20.61134939, 19.46946518, 20.02071643, 20.34514596, 19.79538786,
       20.24498947, 19.23078654, 20.35257698, 21.04011064, 19.34567931,
       17.950876  , 17.65966753, 15.85793459, 15.24560711, 16.65874833,
       16.06310628, 16.02844423, 15.17596104, 16.30717104, 17.11405632,
       15.38434346, 14.70405961, 16.91689234, 17.17063399, 15.29347108,
       14.99877943])
    
    rc = ResultContainer(score=1.0)
    simulated_final_state = simulation_data.eval_data[0].output_data[-1]
    rc.final_state_errors = [simulated_final_state[i] - expected_final_state[i] for i in np.arange(0, len(simulated_final_state))]
    rc.success = np.allclose(expected_final_state, simulated_final_state, rtol=0, atol=1e-2)
    
    return rc