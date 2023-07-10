import os
import numpy as np

from stable_baselines3 import PPO
import gymnasium as gym

from ipydex import IPS

env = gym.make('CartPole-v1', render_mode=None)

IPS()
model = PPO(policy="MlpPolicy", env=env)

model.learn(30000)

path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_data", "model.h5")
print(path)
model.save(path)

env = gym.make('CartPole-v1', render_mode="human")
obs_list = []
obs, _ = env.reset()
done = trunc = False
for i in range(300):
    obs_list.append(obs)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        obs, _ = env.reset()

IPS()