# import trajectory class and necessary dependencies
import sys
from pytrajectory import TransitionProblem, log
import numpy as np
from sympy import cos, sin
import os

import matplotlib.pyplot as plt


class SolutionData():
    pass


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics
    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)
    :return:        xdot
    """
    x1, x2, x3, x4 = xx
    u1, = uu
    
    m = 1.0             # masses of the rods [m1 = m2 = m]
    l = 0.5             # lengths of the rods [l1 = l2 = l]
    
    I = 1/3.0*m*l**2    # moments of inertia [I1 = I2 = I]
    g = 9.81            # gravitational acceleration
    
    lc = l/2.0
    
    d11 = m*lc**2+m*(l**2+lc**2+2*l*lc*cos(x1))+2*I
    h1 = -m*l*lc*sin(x1)*(x2*(x2+2*x4))
    d12 = m*(lc**2+l*lc*cos(x1))+I
    phi1 = (m*lc+m*l)*g*cos(x3)+m*lc*g*cos(x1+x3)

    ff = np.array([     x2,
                        u1,
                        x4,
                -1/d11*(h1+phi1+d12*u1)
                ])
    
    return ff



def solve(problem_spec):
    # system state boundary values for a = 0.0 [s] and b = 2.0 [s]
    xa = problem_spec.xx_start
    xb = problem_spec.xx_end

    T_end = problem_spec.T_transition

    # constraints dictionary
    con = problem_spec.constraints

    ua = problem_spec.u_start
    ub = problem_spec.u_end
    
    first_guess = problem_spec.first_guess

    # create the trajectory object
    S = TransitionProblem(f, a=0.0, b=T_end, xa=xa, xb=xb, ua=ua, ub=ub, constraints=con, use_chains=True, first_guess=first_guess)

    # alter some method parameters to increase performance
    S.set_param('su', 10)

    # start
    x, u = S.solve()

    solution_data = SolutionData()
    solution_data.x_func = x
    solution_data.u_func = u

    save_plot(problem_spec, solution_data)

    return solution_data

def save_plot(problem_spec, solution_data):
    tt = np.linspace(0, problem_spec.T_transition, 1000)

    uu = np.array([solution_data.u_func(t)[0] for t in tt])
    xx = np.array([solution_data.x_func(t) for t in tt])

    plt.figure(figsize=(8,8))
    ax1 = plt.subplot(211)
    plt.plot(tt, xx[:, 0], label=r"$\theta_2$ [rad]")
    plt.plot(tt, xx[:, 1], label=r"$\dot \theta_2$ [rad/s]")
    plt.plot(tt, xx[:, 2], label=r"$\theta_1$ [rad]")
    plt.plot(tt, xx[:, 3], label=r"$\dot \theta_1$ [rad/s]")
    plt.legend()
    plt.ylabel('state')

    plt.subplot(212, sharex=ax1)
    plt.plot(tt, uu)
    plt.ylabel(r"input | $\ddot \theta_2$ [rad/s²]")
    plt.xlabel('$t$ [s]')

    sol_dir = os.path.join(os.path.dirname(__file__), '_solution_data')

    if not os.path.isdir(sol_dir):
        os.mkdir(sol_dir)
    
    plt.savefig(os.path.join(sol_dir, 'plot.pdf'))