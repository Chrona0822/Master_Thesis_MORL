"""
Envelope Q-learning (Yang et al., 2019).

Maintains a Q-vector table Q[w][s, a] ∈ R^d for each weight vector w in a
fixed simplex grid — like TabularGIP but with the envelope Bellman backup:

    Q^w(s,a) ← Q^w(s,a) + lr * (r + γ·Q^{w*}(s',a*) − Q^w(s,a))

where  (w*, a*) = argmax_{w', a'} w^T Q^{w'}(s', a')

Instead of each Q-table bootstrapping only from itself (as in TabularGIP),
the envelope operator borrows the best achievable next-state value from ANY
grid Q-table. This allows information to flow across preference vectors and
produces a tighter approximation of the full Pareto front.

Implementation note:
    Q is stored as a single (n_grid, n_states, n_actions, n_obj) array so
    that both the envelope backup and the TD update can be fully vectorised
    with numpy einsum and fancy indexing — no Python loops over grid points.
    This is critical for the 3-obj case where n_grid = 66.

Reference:
    Yang et al. (2019). "A Generalized Algorithm for Multi-Objective
    Reinforcement Learning and Policy Adaptation." NeurIPS 2019.
"""

import itertools
import numpy as np


def _build_grid(n_obj, n_points=11):
    """Uniform simplex grid — same construction as in tabular_agent.py."""
    ticks = np.linspace(0, 1, n_points)
    grid  = []
    for combo in itertools.product(ticks, repeat=n_obj):
        if abs(sum(combo) - 1.0) < 1e-9:
            grid.append(np.array(combo, dtype=np.float64))
    return grid


class EnvelopeQAgent:
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
        self.n_actions = n_actions
        self.n_obj     = n_obj
        self.gamma     = gamma
        self.lr        = lr
        self.eps       = eps_start
        self.eps_end   = eps_end

        self.grid   = _build_grid(n_obj, n_grid_points)
        self.n_grid = len(self.grid)
        # W[i] = i-th simplex weight vector, shape (n_grid, n_obj)
        self.W = np.array(self.grid, dtype=np.float64)

        # Single contiguous array: Q[i, s, a, d]
        # Avoids per-step Python list indexing; enables numpy vectorisation.
        self.Q = np.zeros((self.n_grid, n_states, n_actions, n_obj),
                          dtype=np.float64)

        self._step_count      = 0
        self._eps_decay_steps = None

    def _state_index(self, obs):
        return int(obs[0]) * 11 + int(obs[1])

    def _envelope_action(self, s, beta):
        """
        Vectorised envelope greedy: argmax_{j, a} beta^T Q[j, s, a].

        scores[j, a] = beta @ Q[j, s, a]  — computed in one einsum call.
        Returns the action a from the globally best (j, a) pair.
        """
        # Q[:, s, :, :] shape: (n_grid, n_actions, n_obj)
        scores = np.einsum('d,jad->ja', beta, self.Q[:, s, :, :])  # (n_grid, n_actions)
        best_flat = int(np.argmax(scores))
        return best_flat % self.n_actions

    def select_action(self, obs, beta):
        if np.random.random() < self.eps:
            return np.random.randint(self.n_actions)
        return self._envelope_action(self._state_index(obs), beta)

    def update(self, obs, action, reward_vec, next_obs, done):
        s  = self._state_index(obs)
        s2 = self._state_index(next_obs)
        r  = np.array(reward_vec, dtype=np.float64)

        if done:
            # All grid weights share the same terminal target
            targets = np.broadcast_to(r, (self.n_grid, self.n_obj)).copy()
        else:
            # Q_s2[j, a, d] = Q[j, s2, a, d], shape (n_grid, n_actions, n_obj)
            Q_s2 = self.Q[:, s2, :, :]

            # scores[i, j, a] = W[i] @ Q[j, s2, a]
            # einsum: W(i,d) × Q_s2(j,a,d) → scores(i,j,a)
            scores = np.einsum('id,jad->ija', self.W, Q_s2)    # (n_grid, n_grid, n_actions)
            scores_flat = scores.reshape(self.n_grid, -1)        # (n_grid, n_grid*n_actions)

            best_flat = np.argmax(scores_flat, axis=1)           # (n_grid,)
            best_j    = best_flat // self.n_actions              # (n_grid,)
            best_a    = best_flat  % self.n_actions              # (n_grid,)

            # targets[i] = r + gamma * Q[best_j[i], s2, best_a[i]]
            # Fancy indexing: Q_s2[best_j, best_a] → (n_grid, n_obj)
            targets = r + self.gamma * Q_s2[best_j, best_a, :]  # (n_grid, n_obj)

        # Vectorised TD update for ALL grid weights in one numpy operation
        self.Q[:, s, action] += self.lr * (targets - self.Q[:, s, action])

    def decay_epsilon(self, total_steps):
        if self._eps_decay_steps is None:
            self._eps_decay_steps = total_steps
        self._step_count += 1
        frac = min(self._step_count / self._eps_decay_steps, 1.0)
        self.eps = self.eps_end + (1.0 - self.eps_end) * np.exp(-5.0 * frac)

    def act_greedy(self, obs, beta):
        return self._envelope_action(self._state_index(obs), beta)
