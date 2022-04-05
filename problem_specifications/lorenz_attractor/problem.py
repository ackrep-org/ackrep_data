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


class ProblemSpecification(object):
    # system symbols for setting up the ODEs
    # x, y, z = sp.symbols("x, y, z")
  
    
    
    # xx = sp.Matrix([x, y, z])  # states of system
    xx0 = np.array([0.1, 0.1, 0.1])  # initial condition

    tt = times = np.linspace(0, 30, 10000) # vector of times for simulation


    @staticmethod
    def rhs(t, xx):
        """ Right hand side of the ODEs
        :param xx:   system states
        :return:     nonlinear state space
        """
        r = 35  
        b = 2  
        sigma = 20
     

        x, y, z = xx

        # motion of equations
        x_dot = - sigma*x + sigma*y
        y_dot = -x*z + r*x - y
        z_dot = x*y - b*z


        ff = np.array([x_dot,
                        y_dot,
                        z_dot])

        return ff

def evaluate_solution(solution_data):
    """
    
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    target_states = [-0.522566539750587, -0.830457089853563, 14.033163222999248]
    success = all(abs(solution_data.res.y[i][-1] - target_states[i]) < 1e-2 for i in np.arange(0,3))
    return ResultContainer(success=success, score=1.0)
