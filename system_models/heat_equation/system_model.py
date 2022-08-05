import sympy as sp
import symbtools as st
import importlib
import sys, os
import pyinduct as pi
import pyinduct.parabolic as parabolic
import numpy as np
from ipydex import IPS, activate_ips_on_exception

from ackrep_core.system_model_management import GenericModel, import_parameters


class Model:

    # Import parameter_file
    params = import_parameters()
    a2 = [float(i[1]) for i in params.get_default_parameters().items()][0]

    n_fem = 17
    l = 1
    T = 1
    
    # start and end of input trajectory
    y0 = -1
    y1 = 4

    # coefs of pde
    coefs = [1, 0, 0, None, None]
    # or try these:
    # coefs = [1, -0.5, -8, None, None]   #  :)))
    _, a1, a0, _, _ = coefs


    temp_domain = pi.Domain(bounds=(0, T), num=100)
    spat_domain = pi.Domain(bounds=(0, l), num=n_fem * 11)

    # initial and test functions
    nodes = pi.Domain(spat_domain.bounds, num=n_fem)
    fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
    act_fem_base = pi.Base(fem_base[-1])
    not_act_fem_base = pi.Base(fem_base[1:-1])
    vis_fems_base = pi.Base(fem_base)

    pi.register_base("act_base", act_fem_base)
    pi.register_base("sim_base", not_act_fem_base)
    pi.register_base("vis_base", vis_fems_base)

    # trajectory
    u = parabolic.RadFeedForward(
        l, T, param_original=coefs, bound_cond_type="dirichlet", actuation_type="dirichlet", y_start=y0, y_end=y1
    )

    # weak form
    x = pi.FieldVariable("sim_base")
    x_dt = x.derive(temp_order=1)
    x_dz = x.derive(spat_order=1)
    phi = pi.TestFunction("sim_base")
    phi_dz = phi.derive(1)
    act_phi = pi.ScalarFunction("act_base")
    act_phi_dz = act_phi.derive(1)
    # weak formulation of the PDE equations (see documentation)
    weak_form = pi.WeakFormulation(
        [
            # ... of the homogeneous part of the system
            pi.IntegralTerm(pi.Product(x_dt, phi), limits=spat_domain.bounds),
            pi.IntegralTerm(pi.Product(x_dz, phi_dz), limits=spat_domain.bounds, scale=a2),
            pi.IntegralTerm(pi.Product(x_dz, phi), limits=spat_domain.bounds, scale=-a1),
            pi.IntegralTerm(pi.Product(x, phi), limits=spat_domain.bounds, scale=-a0),
            # ... of the inhomogeneous part of the system
            pi.IntegralTerm(pi.Product(pi.Product(act_phi, phi), pi.Input(u, order=1)), limits=spat_domain.bounds),
            pi.IntegralTerm(
                pi.Product(pi.Product(act_phi_dz, phi_dz), pi.Input(u)), limits=spat_domain.bounds, scale=a2
            ),
            pi.IntegralTerm(pi.Product(pi.Product(act_phi_dz, phi), pi.Input(u)), limits=spat_domain.bounds, scale=-a1),
            pi.IntegralTerm(pi.Product(pi.Product(act_phi, phi), pi.Input(u)), limits=spat_domain.bounds, scale=-a0),
        ],
        name="main_system",
    )

    # system matrices \dot x = A x + b0 u + b1 \dot u
    cf = pi.parse_weak_formulation(weak_form)
    ss = pi.create_state_space(cf)

    a_mat = ss.A[1]
    b0 = ss.B[0][1]
    b1 = ss.B[1][1]

    # transformation into \dot \bar x = \bar A \bar x + \bar b u
    a_tilde = np.diag(np.ones(a_mat.shape[0]), 0)
    a_tilde_inv = np.linalg.inv(a_tilde)

    a_bar = (a_tilde @ a_mat) @ a_tilde_inv
    b_bar = a_tilde @ (a_mat @ b1) + b0
