"""
Experiment 1 — Two-objective core benchmark.

Trains all three methods on the concave DST with 2 objectives and
evaluates them on 21 evenly-spaced preference vectors.

Saved results (per method, per seed):
  results/exp1/<method>/seed_<seed>_train_returns.npy
  results/exp1/<method>/seed_<seed>_eval_vecs.npy      shape (21, 2)
  results/exp1/<method>/seed_<seed>_eval_scalars.npy   shape (21,)
  results/exp1/<method>/seed_<seed>_hv.npy             scalar

Summary files aggregate across seeds:
  results/exp1/<method>/summary.npz
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mo_gymnasium as mo_gym
from scipy import stats

from agents.dqn_agent    import CondDQNAgent
from agents.tabular_agent import TabularGIPAgent
from agents.pareto_agent  import ParetoQAgent
from experiments.train_utils import train_dqn, train_tabular
from evaluation.metrics import (
    evaluate_agent, hypervolume, eval_beta_grid_2obj
)

SEEDS      = [42, 123, 256, 512, 1024]
N_EPISODES = 5_000
N_EVAL     = 30            # greedy episodes per beta at evaluation time
N_STATES   = 11 * 11

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp1")
EVAL_BETAS = eval_beta_grid_2obj(n=21)

# Reference point for hypervolume (below worst achievable reward)
HV_REF = np.array([-1.0, -210.0])


def _make_env():
    return mo_gym.make("deep-sea-treasure-concave-v0", float_state=True)


def _run_method(name, make_agent_fn, train_fn, seeds):
    method_dir = os.path.join(RESULT_DIR, name)
    os.makedirs(method_dir, exist_ok=True)

    all_train = []
    all_scalars = []
    all_hvs     = []

    for seed in seeds:
        np.random.seed(seed)
        env   = _make_env()
        agent = make_agent_fn()

        train_returns = train_fn(agent, env, N_EPISODES, n_obj=2, seed=seed)
        env.close()

        # Re-open env for evaluation (avoids state bleed from training)
        eval_env = _make_env()
        vecs, scalars = evaluate_agent(agent, eval_env, EVAL_BETAS,
                                        n_episodes=N_EVAL, seed_offset=seed * 1000)
        eval_env.close()

        hv = hypervolume(vecs, HV_REF)

        np.save(os.path.join(method_dir, f"seed_{seed}_train_returns.npy"),
                np.array(train_returns))
        np.save(os.path.join(method_dir, f"seed_{seed}_eval_vecs.npy"),    vecs)
        np.save(os.path.join(method_dir, f"seed_{seed}_eval_scalars.npy"), scalars)
        np.save(os.path.join(method_dir, f"seed_{seed}_hv.npy"),           np.array([hv]))

        all_train.append(train_returns)
        all_scalars.append(scalars)
        all_hvs.append(hv)

        print(f"  [{name}] seed {seed}: HV={hv:.4f}  "
              f"mean_scalar={np.mean(scalars):.3f}")

    all_train   = np.array(all_train)
    all_scalars = np.array(all_scalars)
    all_hvs     = np.array(all_hvs)

    hv_mean = all_hvs.mean()
    hv_std  = all_hvs.std(ddof=1)
    # 95% CI using t-dist with df=4
    ci = stats.t.ppf(0.975, df=4) * hv_std / np.sqrt(len(seeds))

    np.savez(
        os.path.join(method_dir, "summary.npz"),
        train_returns=all_train,
        eval_scalars=all_scalars,
        hvs=all_hvs,
        hv_mean=hv_mean,
        hv_std=hv_std,
        hv_ci95=ci,
        seeds=np.array(seeds),
        eval_betas=np.array(EVAL_BETAS),
    )
    print(f"  [{name}] HV: {hv_mean:.4f} ± {hv_std:.4f}  (95% CI ±{ci:.4f})")
    return all_hvs, all_scalars


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
            train_dqn,
            seeds,
        )

    if "tabular" in active:
        results["tabular"] = _run_method(
            "tabular",
            lambda: TabularGIPAgent(n_states=N_STATES, n_actions=4, n_obj=2,
                                    n_grid_points=11, gamma=0.99, lr=0.1,
                                    eps_start=1.0, eps_end=0.05),
            train_tabular,
            seeds,
        )

    if "pareto" in active:
        results["pareto"] = _run_method(
            "pareto",
            lambda: ParetoQAgent(n_states=N_STATES, n_actions=4, n_obj=2,
                                 gamma=0.99, lr=0.1,
                                 eps_start=1.0, eps_end=0.05),
            train_tabular,
            seeds,
        )

    # Wilcoxon pairwise tests on HV (only meaningful when ≥2 methods ran)
    _wilcoxon_table(results, seeds)
    print(f"[Exp 1] done — results in {RESULT_DIR}")


def _wilcoxon_table(results, seeds):
    from scipy.stats import wilcoxon
    methods = list(results.keys())
    print("\n  Wilcoxon signed-rank p-values (HV):")
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = results[methods[i]][0]
            b = results[methods[j]][0]
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                p = float("nan")
            print(f"    {methods[i]} vs {methods[j]}: p={p:.4f}")


if __name__ == "__main__":
    run()
