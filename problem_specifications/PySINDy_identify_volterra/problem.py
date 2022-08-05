#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from sympy import cos, sin
from math import pi
from ackrep_core import ResultContainer
import symbtools as st
from scipy.integrate import solve_ivp, odeint

from system_models.lotka_volterra.system_model import Model


class ProblemSpecification(object):

    # data for training
    xx0_train = [5, 1]
    tt_train = np.linspace(0, 10, 5000)
    # data for validation
    xx0_test = [4.6, 0.4]
    tt_test = np.linspace(0, 20, 20000)  

    model = Model()

    # create data for SINDy to identify
    rhs = model.get_rhs_func()

    xx_train = solve_ivp(rhs, [tt_train[0], tt_train[-1]], xx0_train, method="RK45", t_eval=tt_train)
    xx_test = solve_ivp(rhs, [tt_test[0], tt_test[-1]], xx0_test, method="RK45", t_eval=tt_test)
    

def evaluate_solution(solution_data):
    """

    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    target_states = [4.62629796, 3.47466344]
    success = all(abs(solution_data.xx_sim[i, -1] - target_states[i]) < 1e-2 for i in np.arange(len(target_states)))
    return ResultContainer(success=success, score=1.0)
