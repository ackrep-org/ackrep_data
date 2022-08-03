#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""some functions are written in this module to deal with
some problems related to system characteristics"""
import sympy as sp
import numpy as np
import control as ctr
from ipydex import IPS


class System_Property(object):
    def __init__(self):
        self.n_state_func = None  # original nonlinear system functions
        self.n_out_func = None  # original nonlinear output functions
        self.sys_state = None  # states of the system
        self.tau = None  # inputs of the system
        self.eqlbr = None  # equilibrium point
        self.aa = None  # linearized system matrix
        self.bb = None  # linearized input matrix
        self.cc = None  # linearized output matrix
        self.dd = None  # linearized feed forward matrix
        self.sys_ss = None  # state space of the system
        self.aa_j = None  # linearized system matrix in jordan form
        self.bb_j = None  # linearized input matrix in jordan form
        self.cc_j = None  # linearized output matrix in jordan form
        self.sys_ss_j = None  # state space of the system in jordan normal form
        self.tau = None  # inputs of the system
        self.sys_tf = None  # TransferFunction with given parameters
        self.sys_tf_p = None  # TransferFunction with unknown parameters
        self.w = None  # eigenvalues of the system
        self.v = None  # eigenvectors of the system

    def sys_linerazition(self, parameter_values=None):
        """
        using the method jacobian to linearize the nonlinear function
        and translate it in the state space model
        :param: parameter_value: parameter values for substitution
        :return: self.aa: linearized system matrix
        :return: self.bb: linearized input matrix
        :return: self.cc: linearized output matrix
        :return: self.d: linearized feed forward matrix
        """
        """if None in (self.n_state_func, self.n_out_func):
            msg = "The system and output functions must be created first."
            raise ValueError(msg)"""

        if parameter_values is None:
            parameter_values = []

        parameter_values = list(self.eqlbr) + list(parameter_values)

        # linearize the nonlinear system around the equilibrium point
        self.aa = self.n_state_func.jacobian(self.sys_state).subs(parameter_values)
        self.bb = self.n_state_func.jacobian(self.tau).subs(parameter_values)
        self.cc = self.n_out_func.jacobian(self.sys_state).subs(parameter_values)
        self.dd = self.n_out_func.jacobian(self.tau).subs(parameter_values)

        return self.aa, self.bb, self.cc, self.dd

    def sys_description(self, mode=True):
        """if the user has already a linear system, then he can call this function
        for building the state space and TransferFunction directly
        If a system only has the unknown parameters, then it can not be formulated
        in state space

        :param mode: mode:True system with given parameters
                     mode:False system with unknown parameters
        :param parameter_values: given parameters for substitution
        :return:sys_ss: state space of the system
        :return:sys_tf: TransferFunction with given parameters
        :return:sys_tf_p: TransferFunction with unknown parameters
        """

        if None in (self.aa, self.bb, self.cc, self.dd):
            msg = "The system equations must be created first."
            raise ValueError(msg)

        assert mode in (True, False)

        if mode:  # system with given parameters
            self.sys_ss = ctr.ss(self.aa, self.bb, self.cc, self.dd)
            self.sys_tf = ctr.tf(self.sys_ss)
            return self.sys_ss, self.sys_tf

        elif not mode:  # system with unknown parameters
            s = sp.Symbol("s")
            self.sys_tf_p = self.cc * (s * (sp.eye(self.aa.shape[0]) - self.aa) ** -1) * self.bb + self.dd
            return self.sys_tf_p
        else:
            raise ValueError("Unexpected value for parameter `mode`: {}".format(mode))

    def calc_eigenvalue_vec(self):
        """
        :return w: eigenvalues of the system
        :return v: eigenvectors of the system
        """
        if self.aa is None:
            msg = "The system matrix must be created first."
            raise ValueError(msg)

        self.w = self.aa.eigenvals()
        self.v = self.aa.eigenvects()

        return self.w, self.v

    def calc_jordan_normal(self):
        """convert the state space of the system in Jordan normal form
        :return sys_ss_j: system in Jordan normal form
        """
        if None in (self.aa, self.bb, self.cc, self.dd):
            msg = "The system equations must be created first."
            raise ValueError(msg)

        # p: similarity transform V.
        # j: Jordan form
        p, j = self.aa.jordan_form()
        self.aa_j = j
        self.bb_j = p.inv() * self.bb
        self.cc_j = self.cc * p.inv()
        self.sys_ss_j = ctr.ss(j, self.bb_j, self.cc_j, self.dd)

        return self.sys_ss_j
