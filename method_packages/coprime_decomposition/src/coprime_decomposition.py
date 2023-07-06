from dataclasses import dataclass
from typing import Callable, Any, Union

import numpy as np
import sympy as sp
import symbtools as st
from ipydex import Container
from ipydex import IPS  # noqa


@dataclass
class CD_ControllerResult:
    f_func: Callable
    h_func: Callable
    c_coeffs: np.ndarray
    debug: Union[Container, Any]


def coprime_decomposition(n_func, d_func, poles) -> CD_ControllerResult:
    """
    design controller
    :param poles: desired poles for CLCP
    :param n_func: numerator of the transfer function
    :param d_func: denominator of the transfer function
    :return: f_func: numerator of the controller
    :return: h_func: denominator of the controller
    :return: c_coeffs : coefficients of controller
    """
    s = sp.Symbol("s")
    n_coeffs = [float(c) for c in st.coeffs(n_func, s)]  # coefficients of numerator
    d_coeffs = [float(c) for c in st.coeffs(d_func, s)]  # coefficients of denominator

    max_order = max(len(n_coeffs) - 1, len(d_coeffs) - 1)  # maximal order

    # find the minial degree for controller by solving aforementioned inequality constraints
    """
    (1) num_order + den_order + 2 = m + 1 + n (m: maximal order of controller, n: maximal order of original system)
    gard of characteristic polynomial of closed system is m + n, so m + n + 1 scalar equations must be fulfilled
    for comparing coefficients.
    (2) num_order m
    (3) den_order <= m

    function (1) is switched to m -> m = num_order + den_order + 1 - n (4)
    m in (4) is substituted by (2) and (3) ->
    num_order <= num_order + den_order + 1 - n (5)
    den_order <= num_order + den_order + 1 - n (6)
    num_order in (5) and den_order in (6) are eliminated from both sides ->
    den_order >= n - 1 -> den_order - n + 1 >= 0
    num_order >= n - 1 -> num_order - n + 1 >= 0
    """
    x1, x2 = sp.symbols("x1, x2")  # x1, x2 are symbols for num_order and den_order in inequality constraints
    res1 = sp.solve([x1 - max_order + 1 >= 0], [x1])
    res2 = sp.solve([x2 - max_order + 1 >= 0], [x2])
    order_n = (res1.args[0]).args[0]
    order_d = (res2.args[0]).args[0]
    m = max(order_n, order_d)
    order_f1 = order_n + 1
    order_h1 = order_d + 1

    # generate controller function
    f_ceoffs = sp.symbols("f0:%d" % order_f1)
    h_ceoffs = sp.symbols("h0:%d" % order_h1)
    f_poly = sum(f_ceoffs[i] * s**i for i in range(order_f1))
    h_poly = sum(h_ceoffs[i] * s**i for i in range(order_h1))

    # desired denominator of closed loop
    c_func = s - poles[0]
    for i in range(len(poles) - 1):
        c_func = c_func * (s - poles[i + 1])

    f_func1 = sp.expand(f_poly * n_func + h_poly * d_func - c_func)

    # compare coefficients to determine the controller fucntion
    func_list = []
    for i in range(m + max_order + 1):
        func_list.append(f_func1.coeff(s, i))

    c_coeffs = sp.solve(func_list, f_ceoffs + h_ceoffs, dic=True)
    f_func = f_poly.subs(c_coeffs)
    h_func = h_poly.subs(c_coeffs)

    return CD_ControllerResult(f_func, h_func, c_coeffs, None)
