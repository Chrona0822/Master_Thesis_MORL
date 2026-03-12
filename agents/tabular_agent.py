"""
Tabular Q-learning with Greedy Interpolation Policy (GIP).

Follows the DPQ-Learning framework of Vaishnav et al. (2025).

A fixed grid B of preference values is defined upfront.
For each grid point, an independent Q-table Q_beta[s, a] is maintained.
Because Q-learning is off-policy, every transition can update all tables
simultaneously by computing the scalar reward beta^T r for each beta.

At test time, when beta does not fall on a grid point, the GIP linearly
interpolates between the two nearest grid Q-tables to obtain an
approximate Q-function, then acts greedily.

For n_obj > 2 the simplex grid is built by itertools.product over
[0, 0.1, ..., 1.0] filtered to rows that sum to 1. This grows as
O(n_points^(d-1)) — 11 tables for 2-obj, 66 for 3-obj.
"""

import itertools
import numpy as np


def _build_grid(n_obj, n_points=11):
    """Return a list of preference vectors on a uniform simplex grid."""
    ticks = np.linspace(0, 1, n_points)
    grid  = []
    for combo in itertools.product(ticks, repeat=n_obj):
        if abs(sum(combo) - 1.0) < 1e-9:
            grid.append(np.array(combo, dtype=np.float32))
    return grid


class TabularGIPAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        n_obj,
        n_grid_points=11,
        gamma=0.99,
        lr=0.1,
        eps_start=1.0,
        eps_end=0.05,
    ):
        self.n_actions  = n_actions
        self.n_obj      = n_obj
        self.gamma      = gamma
        self.lr         = lr
        self.eps        = eps_start
        self.eps_end    = eps_end

        self.grid = _build_grid(n_obj, n_grid_points)
        # Q[i][s, a] for the i-th grid point
        self.Q = [np.zeros((n_states, n_actions), dtype=np.float64) for _ in self.grid]

        self._step_count     = 0
        self._eps_decay_steps = None

    def _state_index(self, obs):
        # obs is (x, y) in an 11×11 grid → single integer
        return int(obs[0]) * 11 + int(obs[1])

    def select_action(self, obs, beta):
        if np.random.random() < self.eps:
            return np.random.randint(self.n_actions)
        return self._greedy_action(obs, beta)

    def _greedy_action(self, obs, beta):
        q_interp = self._interpolate(obs, beta)
        return int(np.argmax(q_interp))

    def _interpolate(self, obs, beta):
        """
        For 2-obj: find two nearest grid points and interpolate.
        For n-obj: find the single nearest grid point (nearest-neighbour
        fallback — see note below).
        """
        if self.n_obj == 2:
            # beta is a scalar beta[0]; beta[1] = 1 - beta[0]
            b = float(beta[0])
            grid_vals = [float(g[0]) for g in self.grid]
            lower = max((v for v in grid_vals if v <= b + 1e-9), default=grid_vals[0])
            upper = min((v for v in grid_vals if v >= b - 1e-9), default=grid_vals[-1])

            i_lo = grid_vals.index(lower)
            i_hi = grid_vals.index(upper)

            if i_lo == i_hi:
                return self.Q[i_lo][self._state_index(obs)]

            rho = (upper - b) / (upper - lower)
            s = self._state_index(obs)
            return rho * self.Q[i_lo][s] + (1 - rho) * self.Q[i_hi][s]
        else:
            # For higher dimensions: nearest-neighbour on simplex
            dists = [np.linalg.norm(beta - g) for g in self.grid]
            i     = int(np.argmin(dists))
            return self.Q[i][self._state_index(obs)]

    def update(self, obs, action, reward_vec, next_obs, done):
        s  = self._state_index(obs)
        s2 = self._state_index(next_obs)

        for i, beta in enumerate(self.grid):
            scalar_r = float(beta @ reward_vec)
            best_next = np.max(self.Q[i][s2]) if not done else 0.0
            td_target = scalar_r + self.gamma * best_next
            self.Q[i][s, action] += self.lr * (td_target - self.Q[i][s, action])

    def decay_epsilon(self, total_steps):
        if self._eps_decay_steps is None:
            self._eps_decay_steps = total_steps
        self._step_count += 1
        frac = min(self._step_count / self._eps_decay_steps, 1.0)
        self.eps = self.eps_end + (1.0 - self.eps_end) * np.exp(-5.0 * frac)

    def act_greedy(self, obs, beta):
        return self._greedy_action(obs, beta)
