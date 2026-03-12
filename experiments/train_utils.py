"""
Generic training loops shared across experiments.

train_dqn      : trains a CondDQNAgent
train_tabular  : trains a TabularGIPAgent or ParetoQAgent
Both return a list of per-episode scalarised returns (for the beta used
that episode), which is used to draw training curves.
"""

import numpy as np
from agents.dqn_agent import sample_beta


def train_dqn(agent, env, n_episodes, n_obj, seed=None):
    """
    Train the preference-conditioned DQN.

    A new beta is sampled from Dir(1) at the start of each episode.
    epsilon is decayed over the total number of environment steps
    (estimated as n_episodes * avg_episode_length; we use n_episodes * 50
    as a conservative estimate, the exact value does not matter much).
    """
    rng = np.random.default_rng(seed)
    total_steps_estimate = n_episodes * 50
    agent.eps_decay_steps = total_steps_estimate

    episode_returns = []

    for ep in range(n_episodes):
        beta       = sample_beta(n_obj, rng)
        obs, _     = env.reset(seed=int(rng.integers(1e6)))
        obs        = np.array(obs, dtype=np.float32)
        cumulative = np.zeros(n_obj, dtype=np.float64)
        done       = False

        while not done:
            action = agent.select_action(obs, beta)
            next_obs, reward_vec, terminated, truncated, _ = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done     = terminated or truncated

            agent.store(obs, beta, action, reward_vec, next_obs, done)
            agent.update()
            agent.decay_epsilon(total_steps_estimate)

            cumulative += reward_vec
            obs         = next_obs

        episode_returns.append(float(beta @ cumulative))

    return episode_returns


def train_tabular(agent, env, n_episodes, n_obj, seed=None):
    """
    Train a tabular agent (Tabular+GIP or Pareto Q).

    Unlike DQN, the tabular agents update all preference tables at each
    transition (they don't need beta sampled per episode — the update
    loop handles all grid points simultaneously). We still pick a beta
    for the behavioural policy (exploration) using a uniform grid sweep.
    """
    rng = np.random.default_rng(seed)
    total_steps_estimate = n_episodes * 50
    beta_list = _uniform_betas(n_obj)

    episode_returns = []

    for ep in range(n_episodes):
        beta   = beta_list[ep % len(beta_list)]
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        obs    = np.array(obs, dtype=np.float32)
        cumulative = np.zeros(n_obj, dtype=np.float64)
        done   = False

        while not done:
            action = agent.select_action(obs, beta)
            next_obs, reward_vec, terminated, truncated, _ = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done     = terminated or truncated

            agent.update(obs, action, reward_vec, next_obs, done)
            agent.decay_epsilon(total_steps_estimate)

            cumulative += reward_vec
            obs         = next_obs

        episode_returns.append(float(beta @ cumulative))

    return episode_returns


def _uniform_betas(n_obj, n=11):
    """Return a list of uniformly spaced betas on the simplex, used for tabular policy."""
    import itertools
    ticks = np.linspace(0, 1, n)
    betas = []
    for combo in itertools.product(ticks, repeat=n_obj):
        if abs(sum(combo) - 1.0) < 1e-9:
            betas.append(np.array(combo, dtype=np.float32))
    return betas
