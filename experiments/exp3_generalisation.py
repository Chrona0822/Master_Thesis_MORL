"""
Experiment 3 — Generalisation across the preference simplex.

Training uses a *sparse* set of beta vectors (the 5 corner/axis-aligned
points).  Evaluation is split into:
  - seen_betas   : the exact betas used during training
  - unseen_betas : a denser grid excluding the training betas

The generalisation gap is computed per method as:
    gap = |mean_seen - mean_unseen| / (|mean_seen| + eps)

Only the DQN is expected to generalise; the tabular/Pareto methods are
included as a comparison baseline for the gap metric.

Results saved under results/exp3/<method>/.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mo_gymnasium as mo_gym
from scipy import stats

from agents.dqn_agent     import CondDQNAgent, sample_beta
from agents.tabular_agent import TabularGIPAgent
from agents.pareto_agent  import ParetoQAgent
from experiments.train_utils import train_tabular
from evaluation.metrics import evaluate_agent, hypervolume, generalisation_gap

SEEDS      = [42, 123, 256, 512, 1024]
N_EPISODES = 5_000
N_EVAL     = 30
N_STATES   = 11 * 11

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp3")
HV_REF     = np.array([-1.0, -210.0])

# 5 sparse training betas (corners + centre)
SEEN_BETAS = [
    np.array([1.0, 0.0], dtype=np.float32),
    np.array([0.75, 0.25], dtype=np.float32),
    np.array([0.5, 0.5], dtype=np.float32),
    np.array([0.25, 0.75], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
]

# Dense evaluation grid, then split into seen / unseen
_full_grid = [np.array([b, 1.0 - b], dtype=np.float32) for b in np.linspace(0, 1, 21)]

def _is_seen(beta):
    return any(np.allclose(beta, s, atol=1e-4) for s in SEEN_BETAS)

UNSEEN_BETAS = [b for b in _full_grid if not _is_seen(b)]
ALL_BETAS    = _full_grid


def _make_env():
    return mo_gym.make("deep-sea-treasure-concave-v0", float_state=True)


def _train_dqn_sparse(agent, env, n_episodes, n_obj, seed=None):
    """
    DQN training but beta is drawn uniformly from SEEN_BETAS only.
    This tests whether the network can still generalise.
    """
    rng = np.random.default_rng(seed)
    total_steps = n_episodes * 50
    agent.eps_decay_steps = total_steps
    returns = []

    for ep in range(n_episodes):
        beta   = SEEN_BETAS[int(rng.integers(len(SEEN_BETAS)))]
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        obs    = np.array(obs, dtype=np.float32)
        cum    = np.zeros(n_obj)
        done   = False

        while not done:
            action = agent.select_action(obs, beta)
            nobs, rvec, term, trunc, _ = env.step(action)
            nobs = np.array(nobs, dtype=np.float32)
            done = term or trunc
            agent.store(obs, beta, action, rvec, nobs, done)
            agent.update()
            agent.decay_epsilon(total_steps)
            cum += rvec
            obs  = nobs

        returns.append(float(beta @ cum))

    return returns


def _run_method(name, make_agent_fn, train_fn, seeds):
    method_dir = os.path.join(RESULT_DIR, name)
    os.makedirs(method_dir, exist_ok=True)

    gaps_seen_all   = []
    gaps_unseen_all = []
    gen_gaps        = []

    for seed in seeds:
        np.random.seed(seed)
        env   = _make_env()
        agent = make_agent_fn()

        train_fn(agent, env, N_EPISODES, n_obj=2, seed=seed)
        env.close()

        eval_env = _make_env()
        _, seen_scalars   = evaluate_agent(agent, eval_env, SEEN_BETAS,
                                            n_episodes=N_EVAL, seed_offset=seed * 1000)
        _, unseen_scalars = evaluate_agent(agent, eval_env, UNSEEN_BETAS,
                                            n_episodes=N_EVAL, seed_offset=seed * 2000)
        eval_env.close()

        gap = generalisation_gap(seen_scalars.mean(), unseen_scalars.mean())
        gen_gaps.append(gap)
        gaps_seen_all.append(seen_scalars.mean())
        gaps_unseen_all.append(unseen_scalars.mean())

        np.save(os.path.join(method_dir, f"seed_{seed}_seen_scalars.npy"),   seen_scalars)
        np.save(os.path.join(method_dir, f"seed_{seed}_unseen_scalars.npy"), unseen_scalars)

        print(f"  [{name}] seed {seed}: seen={seen_scalars.mean():.3f}  "
              f"unseen={unseen_scalars.mean():.3f}  gap={gap:.4f}")

    gen_gaps = np.array(gen_gaps)
    ci = stats.t.ppf(0.975, df=4) * gen_gaps.std(ddof=1) / np.sqrt(len(seeds))

    np.savez(
        os.path.join(method_dir, "summary.npz"),
        gen_gaps=gen_gaps,
        gap_mean=gen_gaps.mean(),
        gap_std=gen_gaps.std(ddof=1),
        gap_ci95=ci,
        seen_means=np.array(gaps_seen_all),
        unseen_means=np.array(gaps_unseen_all),
        seeds=np.array(seeds),
    )
    print(f"  [{name}] gen gap: {gen_gaps.mean():.4f} ± {gen_gaps.std(ddof=1):.4f}  (CI ±{ci:.4f})")
    return gen_gaps


def run(seeds=SEEDS, methods=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    active = set(methods) if methods else {"dqn", "tabular", "pareto"}

    results = {}

    if "dqn" in active:
        results["dqn"] = _run_method(
            "dqn",
            lambda: CondDQNAgent(state_dim=2, n_actions=4, n_obj=2,
                                 gamma=0.99, lr=1e-3,
                                 buffer_capacity=100_000, batch_size=64,
                                 target_sync_every=200,
                                 eps_start=1.0, eps_end=0.05),
            _train_dqn_sparse,
            seeds,
        )

    if "tabular" in active:
        results["tabular"] = _run_method(
            "tabular",
            lambda: TabularGIPAgent(n_states=N_STATES, n_actions=4, n_obj=2,
                                    n_grid_points=11, gamma=0.99, lr=0.1,
                                    eps_start=1.0, eps_end=0.05),
            # pass SEEN_BETAS so tabular trains on the same sparse set as DQN
            lambda agent, env, n_episodes, n_obj, seed=None: train_tabular(
                agent, env, n_episodes, n_obj, seed=seed, beta_list=SEEN_BETAS),
            seeds,
        )

    if "pareto" in active:
        results["pareto"] = _run_method(
            "pareto",
            lambda: ParetoQAgent(n_states=N_STATES, n_actions=4, n_obj=2,
                                 gamma=0.99, lr=0.1,
                                 eps_start=1.0, eps_end=0.05),
            # pass SEEN_BETAS so pareto trains on the same sparse set as DQN
            lambda agent, env, n_episodes, n_obj, seed=None: train_tabular(
                agent, env, n_episodes, n_obj, seed=seed, beta_list=SEEN_BETAS),
            seeds,
        )

    print(f"[Exp 3] done — results in {RESULT_DIR}")


if __name__ == "__main__":
    run()
