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
import symbtools as st

from system_models.pvtol_system.system_model import Model


class ProblemSpecification(object):

    xx0 = np.zeros(6)  # initial condition
    tt = np.linspace(0, 20, 10000) # vector of times for simulation

    model = Model()

    

def evaluate_solution(solution_data):
    """
    
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    target_states = [-44.568209857694654,
                    -3.7059291004860504,
                    44.85003487125722,
                    -43.39618759509638,
                    -0.01029834435828108,
                    -0.06974559900905179]
    success = all(abs(solution_data.res.y[i][-1] - target_states[i]) < 1e-2 for i in np.arange(0,6))
    return ResultContainer(success=success, score=1.0)
