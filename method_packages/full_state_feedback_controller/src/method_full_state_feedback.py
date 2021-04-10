#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""full state feedback method to stabilize an unstable system
by placing the closed-loop poles of a plant in pre-determined
locations in the s-plane.
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
class StateFeedbackResult:
    input_func: Callable
    state_feedback: np.ndarray
    pre_filter: np.ndarray
    debug: Union[Container, Any]


# def state_feedback(system, poles_o, sys_state, yr, debug=False):
def state_feedback(system: tuple, poles_o: list, sys_state: sp.Matrix, eqrt: list, yr, debug=False) -> StateFeedbackResult:
    """
    :param system : tuple (a, b, c, d) of system matrices
    :param poles_o: tuple of closed-loop poles of the system
    :param sys_state: states of nonlinear system
    :param eqrt: equilibrium points of the system
    :param yr: reference output
    :param debug: output control for debugging in unittest(False:normal
    output,True: output local variables and normal output)
    :return: dataclass StateFeedbackResult (controller function, feedback matrix and pre-filter)
    """

    # ignore the PendingDeprecationWarning for built-in packet control
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

    # system matrices
    a = system[0]
    b = system[1]
    c = system[2]
    d = system[3]

    ctr_matrix = ctr.ctrb(a, b)  # controllabilty matrix
    assert np.linalg.det(ctr_matrix) != 0, 'this system is not controllable'

    # full state feedback
    f_t = ctr.acker(a, b, poles_o)

    # pre-filter
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
    # in this case controller function is: -1 * feedback * states + pre-filter * reference output
    # disturbance value ist not considered
    sys_input = -1 * (f_t * small_state)[0] + v[0] * yr + eqrt[len(sys_state)][1]
    input_func = sp.lambdify((sys_state, t), sys_input, modules='numpy')

    # this method returns controller function, feedback matrix and pre-filter
    result = StateFeedbackResult(input_func, f_t, v, None)

    # return internal variables for unittest
    if debug:
        c_locals = Container(fetch_locals=True)
        result.debug = c_locals

    return result
