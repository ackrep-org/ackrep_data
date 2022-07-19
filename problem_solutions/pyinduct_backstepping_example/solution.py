"""
Ackrep solution file for backstepping-based control of an unstable heat equation
"""
from problem import ProblemSpecification
from feedback import (AnalyticBacksteppingController,
                      ApproximatedBacksteppingController)

import matplotlib.pyplot as plt
import matplotlib
import pyinduct as pi
from ackrep_core.system_model_management import save_plot_in_dir

import os

class SolutionData:
    pass


def solve(problem_spec: ProblemSpecification):
    # get values from spec
    spatial_domain = problem_spec.spatial_domain
    orig_params = problem_spec.orig_params
    fem_sys = problem_spec.fem_sys
    modal_sys = problem_spec.modal_sys
    n_modal = problem_spec.n_modal_sim

    # target system parameters (controller parameters)
    # Note: Backstepping is not able to alter a1 or a2!
    a0_t = 0
    tar_params = [problem_spec.a2, problem_spec.a1, a0_t, None, None]

    # define analytic backstepping controller
    analytic_cont = AnalyticBacksteppingController(spatial_domain,
                                                   orig_params,
                                                   fem_sys)

    # define approximated backstepping controllers
    approx_cont_mod = ApproximatedBacksteppingController(orig_params,
                                                         tar_params,
                                                         n_modal,
                                                         spatial_domain,
                                                         modal_sys)
    approx_cont_fem = ApproximatedBacksteppingController(orig_params,
                                                         tar_params,
                                                         n_modal,
                                                         spatial_domain,
                                                         fem_sys)

    sol_data = SolutionData()
    sol_data.u = analytic_cont

    sys = problem_spec.fem_sys.get_system(sol_data.u)
    ics = problem_spec.fem_sys.get_initial_state(problem_spec.initial_profile,
                                            sol_data.u)
    t_sim, q_sim = pi.simulate_state_space(sys, ics, problem_spec.temp_domain)
    x_sim = problem_spec.fem_sys.get_results(q_sim,
                                        sol_data.u,
                                        t_sim,
                                        problem_spec.spatial_domain,
                                        "FEM Simulation")
    
    sol_data.x = x_sim

    # visualization
    # avoid qt since there are problems when running headless in docker container
    matplotlib.use('Agg')
    pi.surface_plot(x_sim)#, title="Surface plots")
    # pi.surface_plot(x_sim, title="Surface plots")

    save_plot_in_dir()

    return sol_data
