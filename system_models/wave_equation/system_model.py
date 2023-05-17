import sympy as sp
import symbtools as st
import importlib
import sys, os
import pyinduct as pi
import numpy as np
from ipydex import IPS, activate_ips_on_exception

from ackrep_core.system_model_management import GenericModel, import_parameters


class TestInput(pi.SimulationInput):
    def _calc_output(self, **kwargs):
        t = kwargs["time"]
        if t < 2 * np.pi:
            u = np.sin(t)
        else:
            u = 0

        return {"output": u}


class Model:
    # Import parameter_file
    params = import_parameters()

    c = [float(i[1]) for i in params.get_default_parameters().items()][0]
    sys_name = "wave equation"

    T = 50
    l = 2 * np.pi
    n = 101
    spat_bounds = (0, l)
    spat_domain = pi.Domain(bounds=spat_bounds, num=n)
    temp_domain = pi.Domain(bounds=(0, T), num=1000)

    init_x = pi.Function(lambda z: 0, domain=spat_bounds)
    init_x_dt = pi.ConstantFunction(0, domain=spat_bounds)

    init_funcs = pi.LagrangeFirstOrder.cure_interval(spat_domain)
    func_label = "init_funcs"
    pi.register_base(func_label, init_funcs, overwrite=True)

    u = TestInput("input")

    x = pi.FieldVariable(func_label)
    phi = pi.TestFunction(func_label)
    weak_form = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x.derive(temp_order=2), phi), spat_bounds, scale=c**2),
            pi.ScalarTerm(pi.Product(pi.Input(u), phi(l)), scale=-1),
            pi.ScalarTerm(pi.Product(x.derive(spat_order=1)(0), phi(0))),
            pi.IntegralTerm(pi.Product(x.derive(spat_order=1), phi.derive(1)), spat_bounds, scale=1),
        ],
        name=sys_name,
    )

    initial_states = {weak_form.name: [init_x, init_x_dt]}
    spatial_domains = {weak_form.name: spat_domain}
    derivative_orders = {weak_form.name: (0, 0)}

    weak_forms = pi.sanitize_input([weak_form], pi.WeakFormulation)
    print("simulate systems: {}".format([f.name for f in weak_forms]))

    print(">>> parse weak formulations")
    canonical_equations = pi.parse_weak_formulations(weak_forms)

    print(">>> create state space system")
    state_space_form = pi.create_state_space(canonical_equations)
