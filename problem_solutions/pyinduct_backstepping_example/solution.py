"""
Ackrep solution file for backstepping-based control of an unstable heat equation
"""
from problem import ProblemSpecification
from feedback import (AnalyticBacksteppingController,
                      ApproximatedBacksteppingController)


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
    # sol_data.u = approx_cont_fem
    # sol_data.u = approx_cont_mod

    return sol_data
