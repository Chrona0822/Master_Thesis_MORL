"""
Experiment 4 — Sensitivity analysis on key hyperparameters.

Sweeps the primary tunable hyperparameter for each method on the 2-obj DST
and reports hypervolume indicator (HV) across 5 seeds.

Sweeps:
    DQN         lr           in {1e-2, 1e-3, 1e-4}
    Tabular GIP n_grid_points in {5, 11, 21}
    Envelope Q  n_grid_points in {5, 11, 21}

Results saved under results/exp4/<method>_<param>/summary.npz.
Each summary.npz contains: hvs, hv_mean, hv_std, hv_ci95, seeds.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mo_gymnasium as mo_gym
from scipy import stats

from agents.dqn_agent      import CondDQNAgent
from agents.tabular_agent  import TabularGIPAgent
from agents.envelope_agent import EnvelopeQAgent
from experiments.train_utils import train_dqn, train_tabular
from evaluation.metrics import evaluate_agent, hypervolume, eval_beta_grid_2obj

SEEDS      = [42, 123, 256, 512, 1024]
N_EPISODES = 5_000
N_EVAL     = 30
N_STATES   = 11 * 11

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp4")
HV_REF     = np.array([-1.0, -210.0])
EVAL_BETAS = eval_beta_grid_2obj(n=21)

# DQN learning rates to sweep
DQN_LRS = [1e-2, 1e-3, 1e-4]

# Grid sizes to sweep for tabular / envelope
GRID_SIZES = [5, 11, 21]


def _make_env():
    return mo_gym.make("deep-sea-treasure-concave-v0", float_state=True)


def _run_config(config_name, make_agent_fn, train_fn, seeds):
    """Run one hyperparameter configuration across all seeds and save results."""
    config_dir = os.path.join(RESULT_DIR, config_name)
    os.makedirs(config_dir, exist_ok=True)

    all_hvs = []

    for seed in seeds:
        np.random.seed(seed)
        env   = _make_env()
        agent = make_agent_fn()

        train_fn(agent, env, N_EPISODES, n_obj=2, seed=seed)
        env.close()

        eval_env = _make_env()
        vecs, _ = evaluate_agent(agent, eval_env, EVAL_BETAS,
                                  n_episodes=N_EVAL, seed_offset=seed * 1000)
        eval_env.close()

        hv = hypervolume(vecs, HV_REF)
        all_hvs.append(hv)

        np.save(os.path.join(config_dir, f"seed_{seed}_hv.npy"), np.array([hv]))

    all_hvs = np.array(all_hvs)
    hv_mean = all_hvs.mean()
    hv_std  = all_hvs.std(ddof=1)
    ci      = stats.t.ppf(0.975, df=4) * hv_std / np.sqrt(len(seeds))

    np.savez(
        os.path.join(config_dir, "summary.npz"),
        hvs=all_hvs,
        hv_mean=hv_mean,
        hv_std=hv_std,
        hv_ci95=ci,
        seeds=np.array(seeds),
    )
    print(f"  [{config_name}] HV: {hv_mean:.4f} ± {hv_std:.4f}  (CI ±{ci:.4f})")
    return all_hvs


def run(seeds=SEEDS, methods=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    active = set(methods) if methods else {"dqn", "tabular", "envelope"}

    # ── DQN: sweep learning rate ───────────────────────────────────────────────
    if "dqn" in active:
        print("\n  DQN — learning rate sweep")
        for lr in DQN_LRS:
            _run_config(
                f"dqn_lr{lr}",
                lambda lr=lr: CondDQNAgent(
                    state_dim=2, n_actions=4, n_obj=2,
                    gamma=0.99, lr=lr,
                    buffer_capacity=100_000, batch_size=64,
                    target_sync_every=200,
                    eps_start=1.0, eps_end=0.05,
                ),
                train_dqn,
                seeds,
            )

    # ── Tabular GIP: sweep grid size ───────────────────────────────────────────
    if "tabular" in active:
        print("\n  Tabular GIP — grid size sweep")
        for g in GRID_SIZES:
            _run_config(
                f"tabular_grid{g}",
                lambda g=g: TabularGIPAgent(
                    n_states=N_STATES, n_actions=4, n_obj=2,
                    n_grid_points=g, gamma=0.99, lr=0.1,
                    eps_start=1.0, eps_end=0.05,
                ),
                train_tabular,
                seeds,
            )

    # ── Envelope Q: sweep grid size ────────────────────────────────────────────
    if "envelope" in active:
        print("\n  Envelope Q — grid size sweep")
        for g in GRID_SIZES:
            _run_config(
                f"envelope_grid{g}",
                lambda g=g: EnvelopeQAgent(
                    n_states=N_STATES, n_actions=4, n_obj=2,
                    n_grid_points=g, gamma=0.99, lr=0.1,
                    eps_start=1.0, eps_end=0.05,
                ),
                train_tabular,
                seeds,
            )

    print(f"\n[Exp 4] done — results in {RESULT_DIR}")


if __name__ == "__main__":
    run()
