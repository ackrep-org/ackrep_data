#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""this module includes two functions, which are used to
transform the matrices and construct the stats space representation.
This module is imported by the module"reduced_observer.py".
"""

import numpy as np
import control as ctr
from ipydex import Container
import sympy as sp


def transformation(system, debug=False):
    """This function is used to divide system, input and
    output matrix into measurable and unmeasurable states
    or renumber them.

    :param system : tuple (a, b, c) of system matrices
    :param debug: output control for debugging in unittest(False:normal
    output,True: output local variables and normal output)
    :return: renumbered matrices a,b,c
    """

    a = np.array(system[0]).copy()
    b = np.array(system[1]).copy()
    c = np.array(system[2]).copy()
    d = np.array(system[3]).copy()

    index_list = []
    # find out which elements in C are equal to one.
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] == 1:
                index_list.append([i, j])

    # exchange row and column
    for [i1, j1] in index_list:
        if i1 != j1:
            c[:, [j1, i1]] = c[:, [i1, j1]]
            a[[[j1, i1]], :] = a[[[i1, j1]], :]
            a[:, [j1, i1]] = a[:, [i1, j1]]
            b[[j1, i1], :] = b[[i1, j1], :]

    if debug:
        c_locals = Container(fetch_locals=True)
        return c_locals

    return a, b, c, d


def observer_gain(rank_q, a_12, a_22, b_1, b_2, poles_o, f_2):
    """This function is used to calculate the observer gain
     for the reduced observer

    :param rank_q : number of reduced observer states
    :param a_12: submatrix of a
    :param a_22: submatrix of a
    :param b_1: submatrix of b
    :param b_2: submatrix of b
    :param poles_o : tuple of complex poles/eigenvalues of the
    observer dynamics
    :param f_2 : state feedback
    :return sys: observer gain
    """

    s = sp.Symbol('s')
    l1, l2 = sp.symbols('l1, l2')
    l_ = sp.Matrix([l1, l2]).reshape(1, 2)

    # characteristic polynomial depending on the free parameters
    eq = sp.det(s * sp.eye(int(rank_q)) - (a_22 - l_ * a_12))

    # desired characteristic polynomial
    des_charpoly = sp.expand((s - poles_o) ** int(rank_q))

    # coefficient comparison: difference polynomial should be
    # identical to the zero polynomial.
    diff_poly = sp.expand(eq - des_charpoly)

    # list of terms of the individual coefficients
    eq_list = []
    for i in range(rank_q):
        # eq_list = []
        eq_list.append(diff_poly.coeff(s, i))
    eq_list = eq_list[0]

    # using the integrating behavior of the controller
    # to solve the two unknown parameters l1 and l2.
    a_k = a_22 - l_ * a_12 - (b_2 - l_ * b_1) * f_2
    res1 = sp.solve([a_k, eq_list], [l1, l2])
    # l_ = l_.subs([(l1, res1[l1]), (l2, res1[l2])])
    l_ = sp.lambdify((l1, l2), l_, modules='numpy')
    l_ = l_(res1[l1], res1[l2])
    return l_


def state_space_func(a, a_11, a_12, a_21, a_22, b, b_1, b_2, c, l_, f_1, f_2):
    """This function is used to construct the status space
    representation for the simulation of the reduced observer
    :param a: submatrix of a
    :param a_11: submatrix of a
    :param a_12: submatrix of a
    :param a_21: submatrix of a
    :param a_22: submatrix of a
    :param b: submatrix of b
    :param b_1: submatrix of b
    :param b_2: submatrix of b
    :param c: submatrix of c
    :param f_1 : state feedback
    :param f_2 : state feedback
    :return sys: the entire system model
    """

    a_k = a_22 - l_ * a_12 - (b_2 - l_ * b_1) @ f_2
    b_k = a_21 - l_ * a_11 + (a_22 - l_ * a_12) \
        * l_ - (b_2 - l_ * b_1) * (f_1 + f_2 * l_)
    c_k = -f_2
    d_k = -1 * (f_1 + f_2 @ l_)
    hyperrow1 = np.hstack([a + b * d_k * c, b * c_k])
    hyperrow2 = np.hstack([b_k * c, a_k])
    a_s = np.vstack([hyperrow1, hyperrow2])
    c_s = np.hstack([l_ * c, np.eye(3)])
    sys = ctr.ss(a_s, np.zeros((a_s.shape[0], 1)), c_s, 0)

    return sys



