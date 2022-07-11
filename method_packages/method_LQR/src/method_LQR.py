#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LQR method to stabilize an unstable system
by using the cost function to choose the optimal poles
for the closed-loop.
"""
from dataclasses import dataclass
from typing import Callable, Any, Union

import control as ctr
import sympy as sp
import numpy as np
from ipydex import Container
import warnings
from ipydex import IPS  # noqa


@dataclass
class LQR_Result:
    input_func: Callable
    state_feedback: np.ndarray
    pre_filter: np.ndarray
    poles_lqr: np.ndarray
    debug: Union[Container, Any]


def lqr_method(system, q_matrix, r_matrix, sys_state, eqrt, yr, debug=False):
    """
    :param system : tuple (a, b, c, d) of system matrices
    :param q_matrix: state weights matrix
    :param r_matrix: input weights matrix
    :param sys_state: states of nonlinear system
    :param eqrt: equilibrium points of the system
    :param yr: reference output
    :param debug: output control for debugging in unittest(False:normal
    output,True: output local variables and normal output)
    :return result: data class LQR_Result (controller function, feedback matrix pre-filter
                    and poles of closed loop)

    """

    # ignore the PendingDeprecationWarning for built-in packet control
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

    # system matrices
    a = system[0]
    b = system[1]
    c = system[2]
    d = system[3]

    # computes the optimal state feedback controller that minimizes the quadratic cost function
    res = ctr.lqr(a, b, q_matrix, r_matrix)

    # full state feedback
    f_t = res[0]
    a1 = a - b * f_t
    v = -1 * (c * a1 ** (-1) * b) ** (-1)  # pre-filter

    '''Since the controller, which is designed on the basis of a linearized system, 
    is a small signal model, the states have to be converted from the large signal model 
    to the small signal model. i.e. the equilibrium points of the original non-linear system must 
    be subtracted from the returned states. 
    
    x' = x - x0 
    x: states of nonlinear system (large signal model)
    x0: equilibrium points
    x': states of controller (small signal model)
    
    And for the same reason, the equilibrium position of the 
    input must be added to controller function.
    '''

    t = sp.Symbol('t')
    # convert states to small signal model
    small_state = sys_state - sp.Matrix([eqrt[i][1] for i in range(len(sys_state))])

    # creat controller function
    # in this case controller function is: -1 * feedback * states + pre-filter * reference output +
    # equilibrium point of input
    # disturbance value ist not considered
    sys_input = -1 * (f_t * small_state)[0] + v[0] * yr + eqrt[len(sys_state)][1]
    input_func = sp.lambdify((sys_state, t), sys_input, modules='numpy')

    poles_lqr = res[2]

    result = LQR_Result(input_func, f_t, v, poles_lqr, None)

    # return internal variables for unittest
    if debug:
        c_locals = Container(fetch_locals=True)
        result.debug = c_locals

    return result
