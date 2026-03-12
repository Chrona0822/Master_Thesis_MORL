"""
Preference-conditioned Deep Q-Network (Cond-DQN).

The network takes [state | beta] as input and outputs Q(s, a; beta) for
all actions. At each episode start, beta is resampled from Dir(1_d) so
the network sees a continuous, uniform distribution over the simplex.

Architecture:  input(state_dim + n_obj)  ->  128 ReLU  ->  128 ReLU  ->  n_actions
Training:      DQN with experience replay and a target network.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)


def sample_beta(n_obj, rng=None):
    """Sample a preference vector uniformly from the (n_obj-1)-simplex via Dir(1)."""
    rng = rng or np.random
    x = rng.exponential(1.0, size=n_obj)
    return (x / x.sum()).astype(np.float32)


class CondDQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        n_obj,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=100_000,
        batch_size=64,
        target_sync_every=200,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=None,
        device=None,
    ):
        self.n_actions = n_actions
        self.n_obj     = n_obj
        self.gamma     = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = state_dim + n_obj
        self.online = QNetwork(input_dim, n_actions).to(self.device)
        self.target = QNetwork(input_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        self.eps            = eps_start
        self.eps_end        = eps_end
        self.eps_decay_steps = eps_decay_steps  # set after knowing total steps
        self._step_count    = 0
        self._grad_steps    = 0

    def _obs_to_tensor(self, obs, beta):
        x = np.concatenate([obs.astype(np.float32), beta])
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, obs, beta):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online(self._obs_to_tensor(obs, beta))
        return int(q.argmax(dim=1).item())

    def store(self, obs, beta, action, reward_vec, next_obs, done):
        scalar_r = float(beta @ reward_vec)
        self.buffer.push(
            obs.astype(np.float32),
            beta,
            action,
            scalar_r,
            next_obs.astype(np.float32),
            float(done),
        )

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs, betas, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_b   = torch.tensor(np.concatenate([obs,      betas], axis=1), dtype=torch.float32, device=self.device)
        nobs_b  = torch.tensor(np.concatenate([next_obs, betas], axis=1), dtype=torch.float32, device=self.device)
        act_b   = torch.tensor(actions, dtype=torch.long,  device=self.device)
        rew_b   = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_b  = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        q_vals = self.online(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(nobs_b).max(dim=1).values
            targets = rew_b + self.gamma * next_q * (1.0 - done_b)

        loss = nn.functional.mse_loss(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def decay_epsilon(self, total_steps):
        """Call once per environment step. Decays eps exponentially."""
        if self.eps_decay_steps is None:
            self.eps_decay_steps = total_steps
        self._step_count += 1
        frac = min(self._step_count / self.eps_decay_steps, 1.0)
        self.eps = self.eps_end + (1.0 - self.eps_end) * np.exp(-5.0 * frac)

    def act_greedy(self, obs, beta):
        """Deterministic greedy action, used during evaluation."""
        with torch.no_grad():
            q = self.online(self._obs_to_tensor(obs, beta))
        return int(q.argmax(dim=1).item())
