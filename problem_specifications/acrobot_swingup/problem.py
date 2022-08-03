import numpy as np
from scipy.integrate import odeint

from ackrep_core import ResultContainer

from ipydex import IPS


class ProblemSpecification(object):
    xx_start = np.array([0.0, 0.0, 3 / 2.0 * np.pi, 0.0])
    xx_end = np.array([0.0, 0.0, 1 / 2.0 * np.pi, 0.0])
    u_start = np.array([0.0])
    u_end = np.array([0.0])
    T_transition = 2.0
    first_guess = {"seed": 1529}
    constraints = {}

    @staticmethod
    def rhs(xx, uu):
        """Right hand side of the vectorfield defining the system dynamics
        :param xx:       state
        :param uu:       input
        :param uuref:    reference input (not used)
        :param t:        time (not used)
        :param pp:       additionial free parameters  (not used)
        :return:        xdot
        """
        if isinstance(xx, np.ndarray):
            # Function should be evaluated numerically
            from numpy import sin, cos
            from numpy import array as vector
        else:
            # Evaluate function symbolically
            from sympy import sin, cos
            from sympy import Matrix as vector

        x1, x2, x3, x4 = xx
        (u1,) = uu

        m = 1.0  # masses of the rods [m1 = m2 = m]
        l = 0.5  # lengths of the rods [l1 = l2 = l]

        I = 1 / 3.0 * m * l**2  # moments of inertia [I1 = I2 = I]
        g = 9.81  # gravitational acceleration

        lc = l / 2.0

        d11 = m * lc**2 + m * (l**2 + lc**2 + 2 * l * lc * cos(x1)) + 2 * I
        h1 = -m * l * lc * sin(x1) * (x2 * (x2 + 2 * x4))
        d12 = m * (lc**2 + l * lc * cos(x1)) + I
        phi1 = (m * lc + m * l) * g * cos(x3) + m * lc * g * cos(x1 + x3)

        ff = vector([x2, u1, x4, -1 / d11 * (h1 + phi1 + d12 * u1)])

        return ff


def evaluate_solution(solution_data):

    # assume solution_data.u_func is callable and returns the desired input trajectory

    P = ProblemSpecification

    def rhs(xx, t):
        u_act = solution_data.u_func(t)
        return P.rhs(xx, u_act)

    tt = np.linspace(0, P.T_transition, 1000)

    xx_res = odeint(rhs, P.xx_start, tt)

    # boolean result
    success = all(abs(xx_res[-1] - P.xx_end) < 1e-2)

    return ResultContainer(success=success)
