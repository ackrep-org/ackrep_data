#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LQR controller design consists of 4 steps:
1. linearize the non-linear system around the equilibrium point.
2. specify weigh matrices
3. calculate state feedback
4. check whether the system have the desired behavior
"""
# try:
#     import method_LQR as mlqr  # noqa
#     import method_system_property as msp  # noqa
# except ImportError:
#     from method_packages.method_LQR import method_LQR as mlqr
#     from method_packages.method_system_property import method_system_property as msp

import matplotlib.pyplot as plt
import symbtools as st
from scipy.integrate import odeint
import sympy as sp
import numpy as np
import os
from ackrep_core.system_model_management import save_plot_in_dir
from stable_baselines3 import PPO
from ipydex import IPS


class SolutionData:
    pass


def solve(problem_spec, kwargs=None):

    env = problem_spec.env

    model = PPO(policy="MlpPolicy", env=env)

    model.learn(30)

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_data", "model.h5")
    print(path)
    model.save(path)

    env.render_mode = "human"
    obs_list = []
    obs, _ = env.reset()
    done = trunc = False
    for i in range(300):
        obs_list.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, _ = env.reset()


    solution_data = SolutionData()
    solution_data.xx = np.array(env.state)  # states of system
    solution_data.res = np.array(obs_list)

    save_plot(problem_spec, solution_data)

    return solution_data


def save_plot(problem_spec, solution_data):
    titles = ["x1", "x2", "x1_dot", "x2_dot"]
    # simulation for LQR
    plt.figure(1)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(np.arange(solution_data.res.shape[0]), solution_data.res[:, i], color="k", linewidth=1)
        plt.grid(1)
        plt.title(titles[i])
        plt.xlabel("time t/s")
        if i == 0:
            plt.ylabel("position [m]")
        elif i == 1:
            plt.ylabel("angular position [rad]")
        elif i == 2:
            plt.ylabel("velocity [m/s]")
        else:
            plt.ylabel("angular velocity [rad/s]")
    plt.tight_layout()
    save_plot_in_dir("plot.png")


