#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We consider the system in cylindrical coordinates. The first time derivatives of the radius and the
orientation are chosen as the new inputs. The system can be decomposed into two subsystems and a differentially flat output
can be easily read off. Based on this partially decoupled system dynamics we use a simple three-staged control law
which drives the system into the origin by aiming for a minimal arc length in the (original) state space.
"""

import sympy as sp
import numpy as np
from ipydex import Container
import warnings
from ipydex import IPS  # noqa

import symbtools as st
import scipy as sc
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt


def cylindical_coordinates(problem_spec):
    """
    backward transformation of input

    :param problem_spec: ProblemSpecification object
    :return: u(x, v)
    """
    x1, x2, x3 = xx = problem_spec.model.xx_symb
    u1, u2 = uu = problem_spec.model.uu_symb
    z1, z2, z3 = problem_spec.zz
    v1, v2 = vv = problem_spec.vv

    z1e = sp.sqrt(x1**2 + x2**2)
    z2e = sp.atan2(x2, x1)
    z3e = x3

    ff = problem_spec.rhs()

    zz_expr = sp.Matrix([z1e, z2e, z3e])

    rplm1 = [(z1e, z1), (sp.expand(-ff[2] / z1**2), v2), (ff[2], -(z1**2) * v2)]

    ffz = zz_expr.jacobian(xx) * ff
    ffz = ffz.subs(rplm1)
    rplm2 = [(ffz[0], v1)]
    ffz = ffz.subs(rplm2)

    vve = vv.subs(st.rev_tuple(rplm1 + rplm2))

    z2e.diff(x1)

    st.lie_deriv(z1e, ff, xx)
    st.lie_deriv(z2e, ff, xx)

    M = vve.jacobian(uu).inverse_ADJ()
    M.simplify()
    M.subs(rplm1)

    uue = M.subs(st.rev_tuple(rplm1[:1])) * vv

    return uue


def shortest_curve(problem_spec):
    """
    Construction of the shortest curve from the z3-axis to the origin

    :param problem_spec: ProblemSpecification object
    :return: interpolation function
    """
    v1, v2 = vv = problem_spec.vv

    r1, dz3, x = sp.symbols("r1, \Delta{}z_3, x", positive=True)

    # Formula for the length of the curve in dependence of Δz3 and r1
    Le = sp.expand(2 * r1 + (r1 * v2) * sp.sqrt(1 + r1**2) * dz3 / r1**2 / v2)

    Led = Le.diff(r1)
    Ledx = Led.subs(r1**2, x)
    sol1, sol2, sol3 = sp.solve(Ledx, x)            # sol = sp.solve(Le.diff(r1), r1) takes too much time
    sol = [sol1**0.5, sol2**0.5, sol3**0.5]

    # Speedup the evaluation later (prepare values for interpolation)
    zz3 = np.logspace(-3, 3, 100)

    rr_opt = [r1_opt(z3_value, sol) for z3_value in zz3]
    r1_opt_interp = sc.interpolate.interp1d(zz3, rr_opt, bounds_error=False, fill_value="extrapolate")

    return r1_opt_interp


def r1_opt(dz3_value, sol):
    """
    selects for every r1-value the real positive solution
    optimal value of r1 in dependence of delta z3

    :param dz3_value: delta z3 value
    :param sol: all solutions
    :return:  real positive solution r1
    """

    r1, dz3 = sp.symbols("r1, \Delta{}z_3", positive=True)

    results = [s.subs(dz3, dz3_value).evalf() for s in sol]

    results.sort(key=lambda x: abs(sp.im(x)))

    r1 = sp.re(results[0])
    if r1 < 0:
        r1 = sp.re(results[1])
    assert r1 >= 0
    return r1


def sigmoid(xmin, xmax, x, slope=40):
    """
    smoothly map x to the range (xmin, xmax)
    This is used to scale the input signals if the tolerance is near (to avoid sliding modes)

    :return: float value
    """
    assert xmax > xmin
    dx = xmax - xmin

    res = xmax - dx / (1 + 0.05 * np.exp(slope * x))
    return res


def cont_continuation(x, stephight=2 * np.pi, threshold=0.01):
    """
    continuous continuation (for 1d-arrays)

    :param x: data
    :param stephight: the expected stephight (e.g 2*pi)
    :param threshold: smallest difference which is considered as a discontinuity which has to be corrected
                    (must be greater than the Lipschitz-Const. of the signal times dt)
    :return: corrected x array
    """
    x_d = np.concatenate(([0], np.diff(x)))
    corrector_array = np.sign(x_d) * (np.abs(x_d) > threshold)
    corrector_array = np.cumsum(corrector_array) * -stephight

    return x + corrector_array


def controller(state, var_controller):
    """
    control law

    :param state: array of initial values
    :param var_controller: list of functions and variables used by the controller
    :return: u1, u2
    """

    x1, x2, x3 = state
    r = np.sqrt(x1**2 + x2**2)
    z = x3
    r1_opt_interp, z_tol, r_tol, vv_to_uu = var_controller
    r_opt_value = r1_opt_interp(abs(z))

    if r == 0:
        # go in x1 direction if phi is not well definded (at r=0)
        return [1, 0]

    if r < r_opt_value and abs(z) >= z_tol:
        # Stage 1
        v1 = 1
        v2 = 0
    elif r >= r_opt_value and abs(z) >= z_tol:
        # Stage 2
        v1 = 0
        v2 = np.sign(z) * sigmoid(0.05, 1, abs(z - z_tol))
    elif abs(z) < z_tol:
        # Stage 3
        v1 = -1
        v2 = 0

    if abs(z) < z_tol and r < r_tol:
        # detection of desired state
        v1 = 0

    # calculate u1, u2, from v1, v2
    return vv_to_uu(v1, v2, x1, x2)


def rhs(var_controller, state, _):
    """
    :param var_controller: list of functions and variables used by the controller
    :param state: array of initial values
    :return: array of caculated state values
    """

    u1, u2 = controller(state, var_controller)
    x1, x2, x3 = state
    return np.array([u1, u2, x2 * u1 - x1 * u2])


def euler(var_controller, rhs, y0, T, dt=0.01):
    """
    euler-forward method for integration

    :param var_controller: list of functions and variables used by the controller
    :param rhs: function
    :param y0: array of initial values
    :param T: simulation time
    :param dt: time step
    :return: list of times, array of simulated state values for one initial state
    """
    res = [y0]
    tt = [0]
    while tt[-1] <= T:
        x_old = res[-1]
        res.append(x_old + dt * rhs(var_controller, x_old, 0))
        tt.append(tt[-1] + dt)
    return tt, np.array(res)


def simulate(var_controller, xx0, Tend, dt):
    """
    simulate by using euler-forward method for integration

    :param var_controller: list of functions and variables used by the controller
    :param xx0: array of initial values
    :param Tend: simulation time
    :param dt: time step
    :return: array of simulated state values for one initial state
    """
    tt, xxn = euler(var_controller, rhs, xx0, Tend, dt=dt)
    return np.array(xxn)
