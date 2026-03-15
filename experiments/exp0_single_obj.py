"""
Experiment 0 — Single-objective convergence check.

Trains the DQN with a fixed beta = [0.5, 0.5] (two objectives, equal
weight) to verify that the DQN implementation learns correctly before
moving to the full multi-objective setup.

Produces: results/exp0/seed_<seed>_returns.npy
          results/exp0/summary.npz  (mean ± std across seeds)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mo_gymnasium as mo_gym

from agents.dqn_agent import CondDQNAgent, sample_beta
from experiments.train_utils import train_dqn

SEEDS       = [42, 123, 256, 512, 1024]
N_EPISODES  = 3_000
FIXED_BETA  = np.array([0.5, 0.5], dtype=np.float32)
RESULT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "exp0")


def run(seeds=SEEDS, methods=None):  # methods ignored; exp0 is DQN-only
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_returns = []

    for seed in seeds:
        np.random.seed(seed)

        env = mo_gym.make("deep-sea-treasure-concave-v0", float_state=True)

        agent = CondDQNAgent(
            state_dim=2, n_actions=4, n_obj=2,
            gamma=0.99, lr=1e-3,
            buffer_capacity=100_000, batch_size=64,
            target_sync_every=200,
            eps_start=1.0, eps_end=0.05,
        )

        # Override beta sampling: always use fixed beta for Exp 0
        def fixed_train(agent, env, n_episodes, n_obj, seed=None):
            from agents.dqn_agent import sample_beta as _sample
            import numpy as _np
            rng = _np.random.default_rng(seed)
            total_steps = n_episodes * 50
            agent.eps_decay_steps = total_steps
            returns = []
            for ep in range(n_episodes):
                beta   = FIXED_BETA
                obs, _ = env.reset(seed=int(rng.integers(1e6)))
                obs    = _np.array(obs, dtype=_np.float32)
                cum    = _np.zeros(n_obj)
                done   = False
                while not done:
                    action = agent.select_action(obs, beta)
                    nobs, rvec, term, trunc, _ = env.step(action)
                    nobs = _np.array(nobs, dtype=_np.float32)
                    done = term or trunc
                    agent.store(obs, beta, action, rvec, nobs, done)
                    agent.update()
                    agent.decay_epsilon(total_steps)
                    cum += rvec
                    obs  = nobs
                returns.append(float(beta @ cum))
            return returns

        returns = fixed_train(agent, env, N_EPISODES, n_obj=2, seed=seed)
        env.close()

        arr = np.array(returns)
        np.save(os.path.join(RESULT_DIR, f"seed_{seed}_returns.npy"), arr)
        all_returns.append(arr)
        print(f"  seed {seed}: final 100-ep mean = {arr[-100:].mean():.3f}")

    all_returns = np.stack(all_returns)   # shape (n_seeds, n_episodes)
    np.savez(
        os.path.join(RESULT_DIR, "summary.npz"),
        returns=all_returns,
        seeds=np.array(SEEDS),
    )
    print(f"[Exp 0] done — results in {RESULT_DIR}")


if __name__ == "__main__":
    run()
