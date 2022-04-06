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


class ProblemSpecification(object):

    xx0 = np.zeros(6)  # initial condition

    tt = np.linspace(0, 20, 10000) # vector of times for simulation

    def uu_func(t, xx):
        return uu_func(t, xx)
    
    @staticmethod
    def rhs(t, xx):
        """ Right hand side of the ODEs
        :param xx:   system states
        :return:     nonlinear state space
        """
        g = 9.81  # m/s^2
        l = 0.1  # m
        m = 0.25 # kg
        J = 0.00076 # kg*m^2

        
        
        
        
        u1, u2 = uu_func(t, xx)
        x1, x2, x3, x4, x5 ,x6 = xx

        # motion of equations
        x1_dot = x2
        x2_dot = -sp.sin(x5)/m * (u1 + u2)
        x3_dot = x4
        x4_dot = sp.cos(x5)/m * (u1 + u2) - g
        x5_dot = x6 *2*sp.pi/360
        x6_dot = l/J * (u2 - u1) *2*sp.pi/360


        ff = np.array([x1_dot,
                        x2_dot,
                        x3_dot,
                        x4_dot,
                        x5_dot,
                        x6_dot])

        return ff

def uu_func(t, xx):
    """
    :param t:(scalar or vector) Time
    :param xx: (vector or array of vectors) state vector with 
                                                numerical values at time t      
    :return:(function with 2 args - t, xx) default input function 
    """ 
    g = 9.81  # m/s^2
    l = 0.1  # m
    m = 0.25 # kg
    J = 0.00076 # kg*m^2

    T_raise = 2
    T_left = T_raise + 2 + 2
    T_right = T_left + 4
    T_straight = T_right + 2
    T_land = T_straight + 3
    force = 0.75*9.81*m
    force_lr = 0.7*9.81*m
    g_nv = 0.5*g*m
    # create symbolic polnomial functions for raise and land
    t_symb = sp.Symbol('t')
    poly1 = st.condition_poly(t_symb, (0, 0, 0, 0), 
                                (T_raise, force, 0, 0))
    
    poly_land = st.condition_poly(t_symb, (T_straight, g_nv, 0, 0), 
                                    (T_land, 0, 0, 0))
    
    # create symbolic piecewise defined symbolic transition functions
    transition_u1 = st.piece_wise((0, t_symb < 0), 
                                    (poly1, t_symb < T_raise), 
                                    (force, t_symb < T_raise + 2), 
                                    (g_nv, t_symb < T_left),
                                    (force_lr, t_symb < T_right),
                                    (g_nv, t_symb < T_straight),
                                    (poly_land, t_symb < T_land),
                                    (0, True))
    
    transition_u2 = st.piece_wise((0, t_symb < 0), 
                                    (poly1, t_symb < T_raise), 
                                    (force, t_symb < T_raise + 2), 
                                    (force_lr, t_symb < T_left),
                                    (g_nv, t_symb < T_right),
                                    (force_lr, t_symb < T_straight),
                                    (poly_land, t_symb < T_land),
                                    (0, True))
    
    # transform symbolic to numeric function 
    transition_u1_func = st.expr_to_func(t_symb, transition_u1)
    transition_u2_func = st.expr_to_func(t_symb, transition_u2)
    
    def uu_rhs(t, xx_nv):
        u1 = transition_u1_func(t)
        u2 = transition_u2_func(t)
                    
        return [u1, u2]
    
    return uu_rhs(t, xx)

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
