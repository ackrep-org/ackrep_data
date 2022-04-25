#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
system description: model description of the lorenz attractor
"""
import numpy as np
import sympy as sp
from sympy import cos, sin
from math import pi
from ackrep_core import ResultContainer
import sys, os
from ipydex import IPS, activate_ips_on_exception  # for debugging only
from system_models.lorenz_system.system_model import Model 

class ProblemSpecification(object):
    xx0 = np.array([0.1, 0.1, 0.1])  # initial condition
    tt = np.linspace(0, 30, 10000) # vector of times for simulation
    
    model = Model()
    rhs = model.get_rhs_func()


def evaluate_solution(solution_data):
    """
    
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    target_states = [-0.522566539750587, -0.830457089853563, 14.033163222999248]
    success = all(abs(solution_data.res.y[i][-1] - target_states[i]) < 1e-2 for i in np.arange(0,3))
    return ResultContainer(success=success, score=1.0)
