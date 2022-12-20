import sympy as sp
import symbtools as st
import importlib
import sys, os
import pyinduct as pi
import numpy as np
from ipydex import IPS, activate_ips_on_exception

from ackrep_core.system_model_management import GenericModel, import_parameters


class Model:
    # Import parameter_file
    params = import_parameters()

    v = [float(i[1]) for i in params.get_default_parameters().items()][0]
    T = 5
    l = 5
    spat_bounds = (0, l)
    spat_domain = pi.Domain(bounds=spat_bounds, num=51)
    temp_domain = pi.Domain(bounds=(0, T), num=100)

    init_x = pi.Function(lambda z: 0, domain=spat_bounds)

    init_funcs = pi.LagrangeFirstOrder.cure_interval(spat_domain)
    func_label = "init_funcs"
    pi.register_base(func_label, init_funcs, overwrite=True)
    # inpuc function
    u = pi.SimulationInputSum(
        [
            pi.SignalGenerator("square", np.array(temp_domain), frequency=0.1, scale=1, offset=1, phase_shift=1),
            pi.SignalGenerator("square", np.array(temp_domain), frequency=0.2, scale=2, offset=2, phase_shift=2),
            pi.SignalGenerator("square", np.array(temp_domain), frequency=0.3, scale=3, offset=3, phase_shift=3),
            pi.SignalGenerator("square", np.array(temp_domain), frequency=0.4, scale=4, offset=4, phase_shift=4),
            pi.SignalGenerator("square", np.array(temp_domain), frequency=0.5, scale=5, offset=5, phase_shift=5),
        ]
    )

    x = pi.FieldVariable(func_label)
    phi = pi.TestFunction(func_label)

    # weak formulation is starting point for calculation (see documentation)
    weak_form = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x.derive(temp_order=1), phi), spat_bounds),
            pi.IntegralTerm(pi.Product(x, phi.derive(1)), spat_bounds, scale=-v),
            pi.ScalarTerm(pi.Product(x(l), phi(l)), scale=v),
            pi.ScalarTerm(pi.Product(pi.Input(u), phi(0)), scale=-v),
        ],
        name=params.model_name,
    )

    ics = pi.sanitize_input(init_x, pi.core.Function)
    initial_states = {weak_form.name: ics}
    spatial_domains = {weak_form.name: spat_domain}
    derivative_orders = {weak_form.name: (0, 0)}

    weak_forms = pi.sanitize_input([weak_form], pi.WeakFormulation)
    print("simulate systems: {}".format([f.name for f in weak_forms]))

    print(">>> parse weak formulations")
    canonical_equations = pi.parse_weak_formulations(weak_forms)

    print(">>> create state space system")
    state_space_form = pi.create_state_space(canonical_equations)
