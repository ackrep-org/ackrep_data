"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union
import os
import numpy as np
from scipy.integrate import solve_ivp
from ipydex import IPS

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from system_models.cartpole_system.system_model import Model


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # meta info
        self.set_name()
        self.seed = None
        self.episode_count = 0
        self.ep_step_count = 0
        self.total_step_count = 0
        self.training = False
        self.history = {
            "step": [],
            "episode": [],
            "state": [],
            "action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "info": [],
        }

        # physics
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        #! pole has length 2*l
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "solve_ivp"  # "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # environment
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,  # x
                np.finfo(np.float32).max,  # xdot
                self.theta_threshold_radians * 2,  # phi
                np.finfo(np.float32).max,  # phidot
            ],
            dtype=np.float32,
        )

        self.action_space = None
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.action = None
        self.reward = None

        # UI variables
        self.target_offset = 0
        self.request_reset = False
        self.reset_button = None  # initialize some objects, that have to be persistent and not be recreated each step
        self.debug_button = None  # initialize some objects, that have to be persistent and not be recreated each step

        self.steps_beyond_terminated = None

    def set_name(self):
        self.name = self.__class__.__name__

    def get_force(self, action):
        raise NotImplementedError("This method has to be overwritten by subclass")

    def calc_new_state(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.get_force(action)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        state = (x, x_dot, theta, theta_dot)
        return state

    def step(self, action):
        self.ep_step_count += 1
        self.total_step_count += 1
        if not self.action_space.contains(action):
            action = np.array([action], dtype=float)
        self.action = action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        old_state = self.state

        self.state = self.calc_new_state(action)

        # Debug reproducibility
        # with open("test.txt", "a") as f:
        #     f.write(f"\nStep. {self.total_step_count}, old State: {old_state}, Action: {action}, new State: {self.state}")

        self.reward, terminated, truncated, info = self.get_reward()

        if self.render_mode == "human":
            self.render()

        # manipulate state to make interactive env with mobile target position
        state = np.array(self.state, dtype=np.float32)
        state[0] -= self.target_offset
        if self.request_reset:
            truncated = True

        self.save_step_data(state, action, self.reward, terminated, truncated, info)

        if hasattr(self, "post_processing_state"):
            state = self.post_processing_state(state)

        return state, self.reward, terminated, truncated, info

    def save_step_data(self, state, action, reward, terminated, truncated, info):
        self.history["step"].append(self.total_step_count)
        self.history["episode"].append(self.episode_count)
        self.history["state"].append(state)
        self.history["action"].append(action)
        self.history["reward"].append(reward)
        self.history["terminated"].append(terminated)
        self.history["truncated"].append(truncated)
        self.history["info"].append(info)

    def get_reward(self):
        x, x_dot, theta, theta_dot = self.state

        truncated = False
        info = {}
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        return reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        state=None,
    ):
        if seed is None and self.seed is not None:
            seed = self.seed + self.episode_count
        super().reset(seed=seed)

        if state is None:
            # random state
            low, high = self.c.get_reset_bounds(self)
            # self.state = self.np_random.uniform(low=low, high=high, size=self.observation_space.shape)
            self.state = self.np_random.uniform(low=low, high=high, size=np.array(low).shape)
        else:
            # fixed state
            self.state = state

        self.steps_beyond_terminated = None
        self.target_offset = 0
        self.request_reset = False

        self.ep_step_count = 0
        self.episode_count += 1

        if self.render_mode == "human":
            self.render()

        s = self.state
        if hasattr(self, "post_processing_state"):
            s = self.post_processing_state(s)

        return np.array(s, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gymnasium[classic-control]`") from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                # TODO does this work if there is only one screen?
                os.environ["SDL_VIDEO_WINDOW_POS"] = f"{2000},{400}"
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # show action
        # if self.action == 0:
        #     gfxdraw.filled_circle(self.surf, int(self.screen_width / 2 - 10), 10, 10, (0, 0, 255))
        # elif self.action == 1:
        #     gfxdraw.filled_circle(self.surf, int(self.screen_width / 2 + 10), 10, 10, (255, 0, 0))

        # flip coordinates
        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            pygame.event.pump()

            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleDiscreteEnv(CartPoleEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        self.action_space = spaces.Discrete(2)


    def get_force(self, action):
        force = self.force_mag if action == 1 else -self.force_mag
        return force

    def get_reward(self):
        return NotImplementedError


class CartPoleContinous2Env(CartPoleEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        b = 0.05
        low = [-0.5, -b, -b, -b]
        high = [0.5, b, b, b]
        self.model = Model()
        self.rhs_model = self.model.get_rhs_func()


        self.action_space = spaces.Box(low, high, (1,), float)

        self.seed = 1

    def get_force(self, action):
        return action

    def calc_new_state(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.get_force(action)

        # based on mathematical pendulum
        def ufunc(t, x):
            return action
        self.model.uu_func = ufunc

        def rhs(t, state):
            x, x_dot, theta, theta_dot = state
            x1, x2, x3, x4 = x, theta, x_dot, theta_dot  # change order
            g = self.gravity
            l = self.length
            m1 = self.masscart
            m2 = self.masspole
            try:
                u1 = force[0]
            except IndexError:
                u1 = force
            # dx1_dt = x3
            # dx2_dt = x4
            # dx3_dt = (-g * m2 * np.sin(2 * x2) / 2 + l * m2 * theta_dot**2 * np.sin(x2) + u1) / (
            #     m1 + m2 * np.sin(x2) ** 2
            # )
            # dx4_dt = (g * (m1 + m2) * np.sin(x2) - (l * m2 * theta_dot**2 * np.sin(x2) + u1) * np.cos(x2)) / (
            #     l * (m1 + m2 * np.sin(x2) ** 2)
            # )

            dx1_dt, dx3_dt, dx2_dt, dx4_dt = self.rhs_model(x1, x2, x3, x4)


            return [dx1_dt, dx3_dt, dx2_dt, dx4_dt]  # change order back

        tt = np.linspace(0, self.tau, 2)
        xx0 = np.array(self.state).flatten()
        s = solve_ivp(rhs, (0, self.tau), xx0, t_eval=tt)

        x, x_dot, theta, theta_dot = s.y[:, -1].flatten()

        state = (x, x_dot, theta, theta_dot)
        return state

    def get_reward(self):
        return NotImplementedError

