"""
Evaluation metrics used across all experiments.

- scalarised_return : beta^T R  for a single episode
- evaluate_agent    : run greedy rollouts over a set of beta vectors,
                      return mean reward vectors and mean scalarised returns
- hypervolume       : HV indicator relative to a reference point
- generalisation_gap: eq. (4) from the thesis
"""

import numpy as np
from scipy.spatial import ConvexHull


def scalarised_return(reward_vec, beta):
    return float(beta @ reward_vec)


def evaluate_agent(agent, env, beta_list, n_episodes=30, seed_offset=0):
    """
    Run `n_episodes` greedy rollouts for each beta in beta_list.

    Returns
    -------
    mean_vecs   : ndarray of shape (len(beta_list), n_obj)
                  mean cumulative reward vector per beta
    mean_scalars: ndarray of shape (len(beta_list),)
                  mean scalarised return per beta
    """
    mean_vecs    = []
    mean_scalars = []

    for beta in beta_list:
        vecs = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_offset + ep)
            obs    = np.array(obs, dtype=np.float32)
            cumulative = np.zeros(len(beta), dtype=np.float64)
            done = False
            while not done:
                action = agent.act_greedy(obs, beta)
                obs, reward_vec, terminated, truncated, _ = env.step(action)
                obs    = np.array(obs, dtype=np.float32)
                cumulative += reward_vec
                done = terminated or truncated
            vecs.append(cumulative)

        vecs = np.array(vecs)
        mean_vecs.append(vecs.mean(axis=0))
        mean_scalars.append(float(beta @ vecs.mean(axis=0)))

    return np.array(mean_vecs), np.array(mean_scalars)


def hypervolume(reward_vectors, reference_point):
    """
    Compute the hypervolume indicator dominated by reward_vectors
    relative to reference_point.

    Uses a simple inclusion–exclusion for 2D and falls back to a
    Monte Carlo estimate for higher dimensions (sufficient for
    statistical comparison between methods).
    """
    pts = np.array(reward_vectors)
    ref = np.array(reference_point)

    if pts.shape[1] == 2:
        return _hv_2d(pts, ref)
    else:
        return _hv_monte_carlo(pts, ref, n_samples=200_000)


def _hv_2d(pts, ref):
    """Exact 2D hypervolume via sweeping the non-dominated front."""
    # Filter points that dominate the reference
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    # Sort by first objective descending
    pts = pts[pts[:, 0].argsort()[::-1]]
    hv = 0.0
    current_best = ref[1]
    for p in pts:
        if p[1] > current_best:
            hv += (p[0] - ref[0]) * (p[1] - current_best)
            current_best = p[1]
    return hv


def _hv_monte_carlo(pts, ref, n_samples=200_000):
    """Monte Carlo HV estimate. Adequate for ranking methods in 3D."""
    upper = pts.max(axis=0)
    samples = np.random.uniform(ref, upper, size=(n_samples, len(ref)))
    dominated = np.any(
        np.all(samples[:, None, :] <= pts[None, :, :], axis=2),
        axis=1,
    )
    vol = np.prod(upper - ref)
    return float(dominated.mean() * vol)


def generalisation_gap(mean_seen, mean_unseen, eps=1e-8):
    """
    Relative difference between mean scalarised return on seen vs unseen
    preference vectors (eq. 5 in the thesis).
    """
    return abs(mean_seen - mean_unseen) / (abs(mean_seen) + eps)


def eval_beta_grid_2obj(n=21):
    """21 evenly-spaced points on the 2-simplex."""
    betas = np.linspace(0, 1, n)
    return [np.array([b, 1.0 - b], dtype=np.float32) for b in betas]


def eval_beta_grid_3obj(step=0.2):
    """Triangular grid with given step size on the 3-simplex (~21 points)."""
    grid = []
    ticks = np.arange(0, 1 + step / 2, step)
    for b0 in ticks:
        for b1 in ticks:
            b2 = 1.0 - b0 - b1
            if b2 >= -1e-9:
                grid.append(np.array([b0, b1, max(b2, 0.0)], dtype=np.float32))
    return grid
