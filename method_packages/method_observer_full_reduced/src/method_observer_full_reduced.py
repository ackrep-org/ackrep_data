#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The program deals with the implementation of the full observer
and the reduced observer. In this code the "specific_observer_funcs"
module is imported for renumbering the matrices that they are in the
form of the reduced observer and for the formation of the state space
of the entire system.
"""
from dataclasses import dataclass
from typing import Callable, Any, Union

import control as ctr
import control.matlab
import numpy as np
import sympy as sp
import specific_observer_funcs as ob_func  # noqa
from ipydex import Container
import warnings
from ipydex import IPS  # noqa

j = 1j  # imaginary unit


@dataclass
class ObserverResult:
    state_feedback: np.ndarray
    observer_gain: np.ndarray
    pre_filter: np.ndarray
    debug: Union[Container, Any]


def full_observer(system, poles_o, poles_s, debug=False) -> ObserverResult:
    """implementation of the complete observer for an autonomous system
    :param system : tuple (a, b, c) of system matrices
    :param poles_o: tuple of complex poles/eigenvalues of the observer
    dynamics
    :param poles_s: tuple of complex poles/eigenvalues of the closed
    system
    :param debug: output control for debugging in unittest(False:normal
    output,True: output local variables and normal output)
    :return: dataclass ObserverResult (controller function, feedback matrix, observer_gain, and pre-filter)
    """

    # systemically relevant matrices
    a = system[0]
    b = system[1]
    c = system[2]
    d = system[3]

    # ignore the PendingDeprecationWarning for built-in packet control
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    ctr_matrix = ctr.ctrb(a, b)  # controllabilty matrix
    assert np.linalg.det(ctr_matrix) != 0, "this system is not controllable"

    # State feedback
    f_t = ctr.acker(a, b, poles_s)

    # pre-filter
    a1 = a - b * f_t
    v = -1 * (c * a1 ** (-1) * b) ** (-1)  # pre-filter

    obs_matrix = ctr.obsv(a, c)  # observability matrix
    assert np.linalg.det(obs_matrix) != 0, "this system is unobservable"

    # calculate observer gain
    l_v = ctr.acker(a.T, c.T, poles_o).T

    # controller function
    # in this case controller function is: -1 * feedback * states + pre-filter * reference output
    # disturbance value ist not considered
    # t = sp.Symbol('t')
    # sys_input = -1 * (f_t * sys_state)[0]
    # input_func = sp.lambdify((sys_state, t), sys_input, modules='numpy')

    # this method returns controller function, feedback matrix, observer gain and pre-filter
    result = ObserverResult(f_t, l_v, v, None)

    # return internal variables for unittest
    if debug:
        c_locals = Container(fetch_locals=True)
        result.debug = c_locals

    return result


def reduced_observer(system, poles_o, poles_s, x0, tt, debug=False):
    """
    :param system: tuple (A, B, C) of system matrices
    :param poles_o: tuple of complex poles/eigenvalues of the
                    observer dynamics
    :param poles_s: tuple of complex poles/eigenvalues of the
                    closed system
    :param x0: initial condition
    :param tt: vector for the time axis
    :param debug: output control for debugging in unittest
                  (False:normal output,True: output local variables
                  and normal output)
    :return yy: output of the system
    :return xx: states of the system
    :return tt: time of the simulation
    """

    # renumber the matrices to construct the matrices
    # in the form of the reduced observer
    a, b, c, d = ob_func.transformation(system, debug=False)

    # row rank of the output matrix
    rank = np.linalg.matrix_rank(c[0])

    # submatrices of the system matrix A
    # and the input matrix
    a_11 = a[0:rank, 0:rank]

    a_12 = a[0:rank, rank : a.shape[1]]
    a_21 = a[rank : a.shape[1], 0:rank]
    a_22 = a[rank : a.shape[1], rank : a.shape[1]]
    b_1 = b[0:rank, :]
    b_2 = b[rank : b.shape[0], :]

    # construct the observability matrix
    # q_1 = a_12
    # q_2 = a_12 * a_22
    # q = np.vstack([q_1, q_2])

    obs_matrix = ctr.obsv(a_22, a_12)  # observability matrix
    rank_q = np.linalg.matrix_rank(obs_matrix)
    assert np.linalg.det(obs_matrix) != 0, "this system is unobservable"

    # ignore the PendingDeprecationWarning for built-in packet control
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    # state feedback
    f_t = ctr.acker(a, b, poles_s)
    c_ = c - d * f_t
    f_1 = f_t[:, 0:1]
    f_2 = f_t[:, 1:]
    l_ = ctr.acker(a_22.T, a_12.T, poles_o).T

    # State representation of the entire system
    # Original system and observer error system,
    # Dimension: 4 x 4)
    sys = ob_func.state_space_func(a, a_11, a_12, a_21, a_22, b, b_1, b_2, c_, l_, f_1, f_2)

    # simulation for the proper movement
    yy, tt, xx = ctr.matlab.initial(sys, tt, x0, return_x=True)
    result = yy, xx, tt, sys

    # return internal variables for unittest
    if debug:
        c_locals = Container(fetch_locals=True)
        return result, c_locals

    return result
