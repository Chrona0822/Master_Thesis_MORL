"""
Three-objective wrapper for DeepSeaTreasure.

The base environment returns [treasure, time_penalty].
This wrapper appends a fuel cost that depends on the agent's row (depth):

    r_fuel(s) = -(1 + d * 0.1)

where d is the row index (0 = surface, 10 = deepest).
This is NOT redundant with the time penalty: time penalty is constant -1
per step, fuel varies with depth, so the 3D Pareto surface contains
trade-offs absent from any pairwise projection.
"""

import numpy as np
import gymnasium as gym


class DSTFuelWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # env.unwrapped traverses the full wrapper chain (e.g. TimeLimit → MO base)
        # to reach the base MO environment that defines reward_space.
        low  = np.append(env.unwrapped.reward_space.low,  -2.1)
        high = np.append(env.unwrapped.reward_space.high,  0.0)
        self.reward_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs is (x, y) — row index is the y component
        row = int(obs[1])
        fuel = -(1.0 + row * 0.1)
        reward = np.append(reward, fuel)
        return obs, reward, terminated, truncated, info
