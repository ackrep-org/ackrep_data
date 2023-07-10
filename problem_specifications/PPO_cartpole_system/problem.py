"""
system description: A cartpole system is considered, which consists of a wagon with the mass M,
a rope with the constant length l, which is attached to the wagon, and a load,
which is located at the free end of the rope. The force that can be impressed on the wagon
is available as a manipulated variable.

problem specification for control problem: design of the LQR controller to to control and
stabilize the x-position of the load.
"""
import numpy as np
import sympy as sp
from sympy import cos, sin, symbols
import gymnasium as gym
from math import pi
from ackrep_core import ResultContainer
from system_models.cartpole_system.system_model import Model

from ipydex import IPS


class ProblemSpecification(object):
    env = gym.make('CartPole-v1', render_mode=None)
    env.render_mode = "human"


def evaluate_solution(solution_data):
    """
    Condition: the x-position of the load reaches 1.5m after 6 seconds at the latest
    :param solution_data: solution data of problem of solution
    :return:
    """
    P = ProblemSpecification
    success = np.sum(np.abs(solution_data.res[:,2][-10])) < 0.1
    return ResultContainer(success=success, score=1.0)
