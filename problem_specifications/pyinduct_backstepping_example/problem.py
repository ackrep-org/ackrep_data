"""
Problem file for ackrep framework
"""
import numpy as np
import pyinduct as pi
from ackrep_core import ResultContainer
from matplotlib import pyplot as plt

from simulation import FEMApproximation, ModalApproximation


class ProblemSpecification:
    """
    Stabilization of an unstable heat equation of the form

    x_dt = a2 x_ddz + a1 x_dz + a0 x

    """
    # original system parameters
    a2 = 1
    a1 = 0
    a0 = 20
    orig_params = [a2, a1, a0, None, None]

    # system/simulation parameters
    z_start = 0
    z_end = 1
    spat_bounds = (z_start, z_end)
    spatial_domain = pi.Domain(bounds=spat_bounds, num=100)
    temp_domain = pi.Domain(bounds=(0, .5), num=100)

    # derive initial profile
    np.random.seed(20210714)
    initial_data = np.random.rand(*spatial_domain.shape)
    initial_profile = pi.Function.from_data(spatial_domain,
                                            initial_data,
                                            domain=spat_bounds)

    # number of basis functions, used for system approximation
    n_fem_sim = 20
    n_modal_sim = 10

    # scenarios to simulate
    fem_sys = FEMApproximation(orig_params, n_fem_sim, spat_bounds)
    modal_sys = ModalApproximation(orig_params, n_modal_sim, spatial_domain)

def evaluate_solution(solution_data):
    sys = ProblemSpecification.fem_sys.get_system(solution_data.u)
    ics = ProblemSpecification.fem_sys.get_initial_state(ProblemSpecification.initial_profile,
                                            solution_data.u)
    t_sim, q_sim = pi.simulate_state_space(sys, ics, ProblemSpecification.temp_domain)
    x_sim = ProblemSpecification.fem_sys.get_results(q_sim,
                                        solution_data.u,
                                        t_sim,
                                        ProblemSpecification.spatial_domain,
                                        "FEM Simulation")
    u_sim = solution_data.u.get_results(t_sim)

    # check the solution
    norm_at_end = np.sum(x_sim.output_data[-1]**2)
    suc = np.isclose(norm_at_end, 0, atol=1e-5)
    rc = ResultContainer(success=suc)

    return rc
