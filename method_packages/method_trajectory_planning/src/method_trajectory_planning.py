#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" trajectory planning for linear and nonlinear system
and follow-up control for stabilizing a unstable system"""

import sympy as sp
import symbtools as st
import numpy as np


class Trajectory_Planning(object):
    def __init__(self, YA, YB, t0, tf, tt):
        self.mod = None  # system model
        self.ff = None  # xd = f(x) + g(x)*u
        self.gg = None
        self.yy = None  # input equation of the system
        self.YA = YA  # initial condition of the input
        self.YB = YB  # final condition of the output
        self.t0 = t0  # start time
        self.tf = tf  # final time
        self.tt = tt  # time axis for simulation
        self.num = None  # numerator of the transfer function
        self.dem = None  # denominator of the transfer function
        self.t = sp.Symbol('t')  # parameter of Polynomial
        self.s = sp.Symbol('s')  # s laplace parameter

    def cal_li_derivative(self):
        """
        Calculate Li derivative to find a direct relationship between the
        output and the input.

        :return: derivatives of the output y until input appears
        """

        yy = sp.Matrix([[self.yy]])

        j = 0
        while True:
            f = sp.Matrix([yy[j]]).jacobian(self.mod.xx)
            f = f * self.ff + f * (self.mod.tau[0] * self.gg)

            # if input appears in yy, than break the loop
            if all(sp.Matrix([yy[j]]).jacobian(self.mod.tau)):
                break

            j += 1
            yy = yy.row_insert(j, f)

        return yy

    def calc_trajectory(self):
        """
        calculate the polynomial trajectory
        :return: polynomial trajectory and it's derivatives
        """
        # state transition
        poly = st.condition_poly(self.t, (self.t0, *self.YA), (self.tf, *self.YB))
        # full transition within the given period
        full_transition = st.piece_wise((self.YA[0], self.t < self.t0),
                                        (poly, self.t < self.tf), (self.YB[0], True))

        poly_d = []
        for i in range(len(self.YA)):
            poly_d.append(full_transition.diff(self.t, i))
        return poly_d

    def num_den_laplace(self, q_poly):
        """
        using the trajectory q(t) combined with numerator and
        denominator of the transfer function to calculate u(t) and y(t)

        :param q_poly: intermediate polynomial q(t)
        :return: u_poly : required input trajectory u(t)
                 y_poly : desired trajectory of the Fe-Ball y(t)
        """

        u_poly = st.do_laplace_deriv(self.dem * q_poly, self.s, self.t)
        y_poly = st.do_laplace_deriv(self.num * q_poly, self.s, self.t)

        return u_poly, y_poly


class Tracking_controller(object):
    """ class for a trajectory controller.
    Attributes:
        yy : actual system output
        xx : system states
        uu : system input
        poles : desired poles of closed loop
        trajectory : desired full transition of the system output
        coefficient : coefficients of error dynamics
        control law : function of the tracking controller
    """

    def __init__(self, yy, xx, uu, poles, trajectory):
        self.yy = yy
        self.xx = xx
        self.uu = uu
        self.poles = poles
        self.trajectory = trajectory
        self.k = sp.symbols('s0:%d' % (len(self.yy) - 1))
        self.coefficient = None
        self.control_law = None

    def error_polynomial(self):
        """
        determine the coefficients of the error dynamics by using full state back
        :param k: all symbolic coefficients
        :param poles: desired poles of the error dynamics
        :return: self.coefficient: coefficients which can stabilize the error dynamics
        """
        s = sp.Symbol('s')
        o = len(self.yy) - 1  # order of error dynamics
        poly = sum((self.k[i] * s ** i for i in range(o)), s ** o)  # generate error dynamics function
        poly_list = []
        for i in range(len(self.poles)):
            poly_list.append(poly.subs(s, self.poles[i]))
        self.coefficient = sp.solve(poly_list, self.k)
        return self.coefficient

    def error_dynamics(self):
        """
        create control law to control a nonlinear system
        :param trajectory: desired trajectory of the system output
        :param yy: output function and it's derivation
        :param xx: system states
        :param uu: system input
        :param k: symbolic coefficients of error dynamics
        :param poles: desired poles of the error dynamics
        :return: control law
        """

        t = sp.Symbol('t')
        coeffs = self.error_polynomial()  # coefficients of error dynamics
        # error dynamics function
        error_poly = sum((-1 * list(coeffs.values())[i]
                          * (self.yy[i] - self.trajectory[i]) for i in
                          range(len(coeffs))), self.trajectory[len(coeffs)])

        # control law
        input_func = self.yy[len(coeffs)] - error_poly
        c_law = (sp.solve(input_func, self.uu)[0])
        # control law to function for simulation
        self.control_law = sp.lambdify((self.xx, t), c_law, modules='numpy')
        return self.control_law, c_law
