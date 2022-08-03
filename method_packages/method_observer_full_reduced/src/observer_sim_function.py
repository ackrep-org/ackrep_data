"""create rhs function for simulation of nonlinear system
with linear full observer"""
import sympy as sp
import symbtools as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from symbtools import modeltools as mt
from sympy import sin, cos


class Observer_SimModel(object):
    def __init__(self, f, g, xx, system_tuple, eqrt, l_v, feedback, v, y_target, model_parameters=None):
        """
        'Constructor' method

        :param f: drift vector field
        :param g: matrix whose columns are the input vector fields
        :param xx: states of system
        :param system_tuple: linearized system in tuple e.g. system_tuple = (a, b, c, d)
        :param eqrt: equilibrium points of system
        :param l_v:  observer gain
        :param feedback: controller gain
        :param v: pre-filter
        :param y_target target value pf output
        :param model_parameters: system parameters
        of the parameters
        """
        self.ff = sp.Matrix(f)  # x_dot = f(x) + g(x) * u
        self.gg = sp.Matrix(g)
        self.xx = xx  # states of system
        self.xx_e = sp.Matrix(sp.symbols("xx0:%d" % (len(self.xx))))  # estimated states from full observer
        self.system = system_tuple  # system tuple of the linearized system
        self.eqrt = eqrt
        self.l_v = sp.Matrix(l_v)  # observer gain
        self.feedback = feedback  # state feedback
        self.v = v  # pre-filter
        self.y_target = y_target  # reference output

        if model_parameters is None:
            self.mod_params = []
        else:
            self.mod_params = model_parameters

        self.state_dim = f.shape[0]  # dimension of states
        self.input_dim = g.shape[1]  # dimension of input

    def _create_input_func(self):
        t = sp.Symbol("t")

        """Since the nonlinear system is a large signal model, the equilibrium point of input 
        must be added to the controller function, which is calculated from the small signal model (observer).
        """

        # nonlinear system input
        # observer input
        state_feedback_small = -1 * (self.feedback * self.xx_e)[0] + self.v[0] * self.y_target
        state_feedback = state_feedback_small + self.eqrt[len(self.xx)][1]

        feedback_func = sp.lambdify((*self.xx_e, t), state_feedback, modules="numpy")
        feedback_func_small = sp.lambdify((*self.xx_e, t), state_feedback_small, modules="numpy")

        return feedback_func, feedback_func_small

    def _recreate_simfunction(self):
        a = sp.Matrix(self.system[0])
        b = sp.Matrix(self.system[1])
        c = sp.Matrix(self.system[2])
        d = sp.Matrix(self.system[3])
        """
        the controller, which is designed on the basis of a linearized system, 
        is a small signal model, the states have to be converted from the large signal model 
        to the small signal model. i.e. the equilibrium points of the original nonlinear system must 
        be subtracted from the returned states.
        
        ff_o: f(x) function of observer
        x_e: estimated states, x: true states, x0: equilibrium points
        ff_o(t) = A * x_e + l_v * c * (x - x0 - x_e)
        """
        ff_o = (
            a * self.xx_e
            + self.l_v * (c * (self.xx - sp.Matrix([self.eqrt[i][1] for i in range(len(self.xx))]) - self.xx_e))[0]
        )

        # combine f(x) of original system and f(x) of observer for simulation
        ff_sim = self.ff.row_insert(self.state_dim, ff_o).subs(self.mod_params)

        # combine g(x) of original system and g(x) of observer for simulation
        gg_sim = self.gg.row_insert(self.state_dim, b).subs(self.mod_params)

        # combine true states and estimated states to new states
        xx_all = list(self.xx.row_insert(self.state_dim, self.xx_e))
        # ff_sim and gg_sim to function
        f_func = sp.lambdify(xx_all, ff_sim, modules="numpy")
        g_func = sp.lambdify(xx_all, gg_sim, modules="numpy")

        return f_func, g_func

    def calc_observer_rhs_func(self):
        f_func, g_func = self._recreate_simfunction()
        u_func_non, u_func_observer = self._create_input_func()
        dim = self.state_dim

        def rhs(xx, time):
            xx = np.ravel(xx)
            # reduced states are estimated states
            uu_non = u_func_non(*xx[dim:], time)  # nonlinear system input -> large signal model
            uu_observer = u_func_observer(*xx[dim:], time)  # observer input -> small signal model
            ff = f_func(*xx)
            gg = g_func(*xx)
            gg_non = np.dot(gg[0:dim], uu_non)
            gg_observer = np.dot(gg[dim:], uu_observer)

            xx_dot = ff + np.vstack((gg_non, gg_observer))

            return np.ravel(xx_dot)

        return rhs
