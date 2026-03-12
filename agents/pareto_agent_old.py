"""
Pareto Q-learning (Van Moffaert et al., 2014).

Instead of scalarizing before learning, this method maintains for each
(state, action) pair a *set* of non-dominated Q-vectors representing the
Pareto front of expected cumulative rewards.

Update rule (eq. in thesis):
    Q(s,a) = ND { r + gamma * q'  |  q' in union_{a'} Q(s', a') }

At execution time, beta is used to pick the action whose Q-set contains
the vector maximising beta^T q.

The non-domination filter ND keeps only vectors where no other vector
in the set is component-wise >= and strictly better in at least one
dimension.
"""

import numpy as np


def _is_dominated(v, others):
    """Return True if v is weakly dominated by any vector in others."""
    for u in others:
        if np.all(u >= v) and np.any(u > v):
            return True
    return False


def _non_dominated(vectors):
    """Return the subset of vectors that are not dominated by any other."""
    vectors = list(vectors)
    nd = []
    for i, v in enumerate(vectors):
        rest = [vectors[j] for j in range(len(vectors)) if j != i]
        if not _is_dominated(v, rest):
            nd.append(v)
    return nd


class ParetoQAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        n_obj,
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

        # Q[s][a] is a list of reward-vectors (numpy arrays)
        # Initialised with a single zero vector per (s, a)
        zero = np.zeros(n_obj, dtype=np.float64)
        self.Q = [[[ zero.copy() ] for _ in range(n_actions)]
                  for _ in range(n_states)]

        self._step_count      = 0
        self._eps_decay_steps = None

    def _state_index(self, obs):
        return int(obs[0]) * 11 + int(obs[1])

    def _best_q_for_beta(self, s, beta):
        """Return the Q-vector in Q[s][a] that maximises beta^T q, for each a."""
        best = []
        for a in range(self.n_actions):
            scores = [float(beta @ q) for q in self.Q[s][a]]
            best.append(self.Q[s][a][int(np.argmax(scores))])
        return best  # list of n_actions vectors

    def select_action(self, obs, beta):
        if np.random.random() < self.eps:
            return np.random.randint(self.n_actions)
        return self._greedy_action(obs, beta)

    def _greedy_action(self, obs, beta):
        s    = self._state_index(obs)
        best = self._best_q_for_beta(s, beta)
        scores = [float(beta @ q) for q in best]
        return int(np.argmax(scores))

    def update(self, obs, action, reward_vec, next_obs, done):
        s  = self._state_index(obs)
        s2 = self._state_index(next_obs)

        # Collect all Q-vectors from all actions in next state
        if done:
            successors = [np.zeros(self.n_obj)]
        else:
            successors = []
            for a in range(self.n_actions):
                successors.extend(self.Q[s2][a])
            successors = _non_dominated(successors)

        # Generate candidate new Q-vectors: r + gamma * q' for each q'
        candidates = [reward_vec + self.gamma * q for q in successors]

        # Merge with existing set and keep non-dominated
        merged = self.Q[s][action] + candidates
        self.Q[s][action] = _non_dominated(merged)

    def decay_epsilon(self, total_steps):
        if self._eps_decay_steps is None:
            self._eps_decay_steps = total_steps
        self._step_count += 1
        frac = min(self._step_count / self._eps_decay_steps, 1.0)
        self.eps = self.eps_end + (1.0 - self.eps_end) * np.exp(-5.0 * frac)

    def act_greedy(self, obs, beta):
        return self._greedy_action(obs, beta)
