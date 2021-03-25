#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calculate the equilibrium points of a system by given input values
for linearization.
"""
import sympy as sp
import symbtools as st
import numpy as np
from scipy.optimize import fmin
from symbtools import modeltools as mt
import ipydex


def calc_eqlbr_rt1(mod, uu, sys_paras, ttheta_guess=None, display=False, debug=False):
    """
     In the simplest case, only one of the equilibrium points of
    a nonlinear system is used for linearization.Such a equilibrium
    point is calculated by minimizing the target function for a certain
    input variable.

    :param mod: symbolic_model
    :param uu: system inputs
    :param sys_paras: system parameters
    :param ttheta_guess: initial guess of the equilibrium points
    :param display: Set to True to print convergence messages.
    :param debug: output control for debugging in unittest(False:normal
    output,True: output local variables and normal output)
    :return: Parameter that minimizes function
    """
    # set all of the derivatives of the system states to zero
    stat_eqns = mod.eqns.subz0(mod.ttd, mod.ttdd)
    all_vars = st.row_stack(mod.tt, mod.uu)

    # target function for minimizing
    mod.stat_eqns_func = st.expr_to_func(all_vars, stat_eqns.subs(sys_paras))

    if ttheta_guess is None:
        ttheta_guess = st.to_np(mod.tt * 0)

    def target(ttheta):
        """target function for the certain global input values uu
        """
        all_vars = np.r_[ttheta, uu]  # combine arrays
        rhs = mod.stat_eqns_func(*all_vars)

        return np.sum(rhs ** 2)

    res = fmin(target, ttheta_guess, disp=display)

    if debug:
        C = ipydex.Container(fetch_locals=True)

        return C, res

    return res


def calc_eqlbr_nl(mod, uu, sys_paras, debug=False):
    """
    using Slovers in Sympy to solve the differential equations
    for certain input values for calculating all of the equilibrium points
    of a nonlinear system

    :param mod: symbolic_model
    :param uu: system inputs
    :param sys_paras: system parameters
    :param debug: output control for debugging in unittest(False:normal
    output,True: output local variables and normal output)
    :return: all of the equilibrium points
    """

    def target(ttheta, debug=False):
        rhs1 = mod.solve_for_acc().subs(sys_paras)

        # substitute the actual inputs with the chosen input values
        uu_para = zip(mod.uu, uu)
        rhs = rhs1.subs(list(uu_para))

        if debug:
            C1 = ipydex.Container(fetch_locals=True)

            return C1
        return sp.solve(rhs, ttheta)

    res = target(mod.tt)

    if debug:
        C = ipydex.Container(fetch_locals=True)

        return C

    return res

