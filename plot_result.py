"""
plot_results.py  —  visualise saved experiment results.

Usage:
    python plot_results.py --exp 0        # training curve for Exp 0
    python plot_results.py --exp 1        # Pareto front + scalarised return for Exp 1
    python plot_results.py --exp 2        # 3-obj results for Exp 2
    python plot_results.py --exp 3        # generalisation gap for Exp 3
    python plot_results.py                # all of the above

Figures are saved to results/figures/.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
from scipy import stats

# ── Publication-quality global style ─────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")

# ── Shared method style registry (avoids duplication across functions) ────────
METHODS = {
    "dqn":     {"label": "Cond-DQN",    "color": "#2563eb", "marker": "o"},
    "tabular": {"label": "Tabular+GIP", "color": "#dc2626", "marker": "s"},
    "pareto":  {"label": "Pareto Q",    "color": "#16a34a", "marker": "^"},
}

# ── Region masks for Exp 1 beta analysis (thesis Section 3.5) ─────────────────
# The 21 evaluation betas have beta_0 = linspace(0, 1, 21)
_BETA0 = np.linspace(0, 1, 21)
REGIONS = {
    "Extreme":      (_BETA0 <= 0.2) | (_BETA0 >= 0.8),   # indices 0-4 and 16-20
    "Intermediate": (_BETA0 >  0.2) & (_BETA0 <  0.8),   # indices 5-15
    "Central":      (_BETA0 >= 0.4) & (_BETA0 <= 0.6),   # indices 8-12
}
# Human-readable x-tick labels for the grouped bar chart
REGION_TICK_LABELS = [
    "Extreme\n(β₀ ≤ 0.2 or ≥ 0.8)",
    "Intermediate\n(0.2 < β₀ < 0.8)",
    "Central\n(0.4 ≤ β₀ ≤ 0.6)",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close()


def _smooth(x, window=50):
    """Simple moving-average smoothing for training curves."""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def _add_region_bands(ax):
    """
    Shade the beta axis with three bands (thesis Section 3.5):
      Extreme      : β₀ ∈ [0, 0.2] ∪ [0.8, 1.0]   (grey)
      Intermediate : β₀ ∈ (0.2, 0.8)                (yellow)
      Central      : β₀ ∈ [0.4, 0.6]                (orange, sub-band of Intermediate)

    Uses a blended transform so x is in data coords and y spans the full axes height.
    """
    ax.axvspan(0.0, 0.2, color="#94a3b8", alpha=0.13, zorder=0)
    ax.axvspan(0.8, 1.0, color="#94a3b8", alpha=0.13, zorder=0)
    ax.axvspan(0.2, 0.8, color="#fde68a", alpha=0.10, zorder=0)
    ax.axvspan(0.4, 0.6, color="#fdba74", alpha=0.18, zorder=0)

    # Text labels: x in data coords, y in axes fraction (0–1)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x_pos, text, color in [
        (0.10, "Extreme",      "#475569"),
        (0.50, "Intermediate", "#78350f"),
        (0.50, "(Central)",    "#c2410c"),
        (0.90, "Extreme",      "#475569"),
    ]:
        y = 0.03 if text != "(Central)" else 0.10
        ax.text(x_pos, y, text, ha="center", va="bottom",
                fontsize=6.5, color=color, style="italic",
                transform=trans)


def _plot_hv_bars(exp_dir, methods, filename, title):
    """Reusable hypervolume bar chart with 95 % CI error bars."""
    labels, means, cis, colors = [], [], [], []

    for mname, minfo in methods.items():
        mdir = os.path.join(exp_dir, mname)
        if not os.path.exists(mdir):
            continue
        d = np.load(os.path.join(mdir, "summary.npz"))
        labels.append(minfo["label"])
        means.append(float(d["hv_mean"]))
        cis.append(float(d["hv_ci95"]))
        colors.append(minfo["color"])

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, means, color=colors, width=0.5, alpha=0.88,
                  yerr=cis, capsize=6, error_kw={"linewidth": 1.3, "ecolor": "#374151"})
    # Value labels on top of bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{mean:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hypervolume indicator  (mean ± 95 % CI)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y")
    fig.tight_layout()
    _savefig(filename)


def _plot_region_analysis(exp_dir, methods, filename):
    """
    Grouped bar chart: mean scalarised return per preference region per method.

    Three regions (thesis Section 3.5):
      Extreme      β₀ ∈ [0, 0.2] ∪ [0.8, 1.0]
      Intermediate β₀ ∈ (0.2, 0.8)
      Central      β₀ ∈ [0.4, 0.6]   (sub-band of Intermediate)

    This directly tests H1: DQN advantage expected to be largest in the
    Intermediate / Central region, where linear interpolation fails most.
    """
    region_keys   = ["Extreme", "Intermediate", "Central"]
    masks         = [REGIONS[k] for k in region_keys]
    method_names  = [m for m in methods if os.path.exists(os.path.join(exp_dir, m))]
    n_methods     = len(method_names)
    bar_width     = 0.22
    x             = np.arange(len(region_keys))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, mname in enumerate(method_names):
        minfo   = methods[mname]
        d       = np.load(os.path.join(exp_dir, mname, "summary.npz"))
        scalars = d["eval_scalars"]   # (n_seeds, 21)

        # Per-region mean and 95 % CI computed across seeds
        region_means = [scalars[:, mask].mean() for mask in masks]
        region_cis   = [
            stats.t.ppf(0.975, df=4)
            * scalars[:, mask].mean(axis=1).std(ddof=1)
            / np.sqrt(scalars.shape[0])
            for mask in masks
        ]

        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, region_means, bar_width,
               color=minfo["color"], label=minfo["label"], alpha=0.88,
               yerr=region_cis, capsize=4,
               error_kw={"linewidth": 1.1, "ecolor": "#374151"})

    ax.axhline(0, color="#374151", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_TICK_LABELS)
    ax.set_ylabel("Mean scalarised return  (95 % CI)")
    ax.set_title("Exp 1 — Regional analysis of preference performance\n"
                 "(Tests H1: DQN advantage expected largest in Intermediate/Central region)")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    _savefig(filename)


# ── Exp 0 ─────────────────────────────────────────────────────────────────────

def plot_exp0():
    """Training curve for Exp 0 — DQN convergence check with fixed β = [0.5, 0.5]."""
    exp_dir  = os.path.join(RESULTS_DIR, "exp0")
    data     = np.load(os.path.join(exp_dir, "summary.npz"))
    returns  = data["returns"]      # (n_seeds, n_episodes)

    smoothed = np.array([_smooth(r) for r in returns])
    mean     = smoothed.mean(axis=0)
    std      = smoothed.std(axis=0, ddof=1)
    x        = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    color = METHODS["dqn"]["color"]
    ax.plot(x, mean, color=color, linewidth=1.8, label="Mean (5 seeds)")
    ax.fill_between(x, mean - std, mean + std,
                    alpha=0.20, color=color, label="±1 std")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Scalarised return  (β = [0.5, 0.5])")
    ax.set_title("Exp 0 — DQN convergence check (fixed preference)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig("exp0_training_curve.png")


# ── Exp 1 ─────────────────────────────────────────────────────────────────────

def plot_exp1():
    """
    Exp 1 — Two-objective benchmark.

    (A) Training curves for all three methods, mean ± 1 std across seeds.
    (B) Scalarised return vs β₀, with shaded preference regions (95 % CI).
    (C) Pareto-front approximation scatter in objective space.

    Separate figures:
        exp1_hypervolume.png  — HV bar chart
        exp1_regions.png      — Regional analysis bar chart (thesis Section 3.5)
    """
    exp_dir = os.path.join(RESULTS_DIR, "exp1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    for mname, minfo in METHODS.items():
        mdir = os.path.join(exp_dir, mname)
        if not os.path.exists(mdir):
            continue
        d = np.load(os.path.join(mdir, "summary.npz"))

        # ── Panel A: training curves ──────────────────────────────────────────
        smoothed = np.array([_smooth(r) for r in d["train_returns"]])
        mean = smoothed.mean(axis=0)
        std  = smoothed.std(axis=0, ddof=1)
        axes[0].plot(np.arange(len(mean)), mean,
                     color=minfo["color"], label=minfo["label"], linewidth=1.6)
        axes[0].fill_between(np.arange(len(mean)), mean - std, mean + std,
                             alpha=0.15, color=minfo["color"])

        # ── Panel B: scalarised return vs β₀ ─────────────────────────────────
        scalars  = d["eval_scalars"]        # (n_seeds, 21)
        betas_b0 = d["eval_betas"][:, 0]
        s_mean   = scalars.mean(axis=0)
        s_std    = scalars.std(axis=0, ddof=1)
        ci       = stats.t.ppf(0.975, df=4) * s_std / np.sqrt(scalars.shape[0])
        axes[1].plot(betas_b0, s_mean,
                     color=minfo["color"], label=minfo["label"],
                     linewidth=1.6, marker=minfo["marker"],
                     markersize=4, markevery=2, markeredgecolor="white",
                     markeredgewidth=0.5)
        axes[1].fill_between(betas_b0, s_mean - ci, s_mean + ci,
                             alpha=0.15, color=minfo["color"])

        # ── Panel C: Pareto-front scatter ─────────────────────────────────────
        vecs_list = [np.load(os.path.join(mdir, sf))
                     for sf in sorted(os.listdir(mdir)) if "eval_vecs" in sf]
        if vecs_list:
            vecs = np.mean(vecs_list, axis=0)   # (21, 2)
            axes[2].scatter(vecs[:, 0], vecs[:, 1],
                            color=minfo["color"], label=minfo["label"],
                            s=45, alpha=0.88, marker=minfo["marker"],
                            edgecolors="white", linewidths=0.5, zorder=3)

    # ── Format Panel A ────────────────────────────────────────────────────────
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Scalarised return")
    axes[0].set_title("(A) Training curves")
    axes[0].legend(loc="lower right")

    # ── Format Panel B — add region bands AFTER data so ylim is correct ───────
    _add_region_bands(axes[1])
    axes[1].set_xlabel("β₀  (weight on treasure objective)")
    axes[1].set_ylabel("Scalarised return")
    axes[1].set_title("(B) Return vs preference  (95 % CI)")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(-0.02, 1.02)

    # ── Format Panel C ────────────────────────────────────────────────────────
    axes[2].set_xlabel("Treasure value")
    axes[2].set_ylabel("Time penalty (cumulative)")
    axes[2].set_title("(C) Pareto-front approximation")
    axes[2].legend(loc="lower right")

    fig.suptitle("Exp 1 — Two-objective benchmark", fontsize=12)
    _savefig("exp1_results.png")

    # ── Separate figures ──────────────────────────────────────────────────────
    _plot_hv_bars(exp_dir, METHODS, "exp1_hypervolume.png",
                  title="Exp 1 — Hypervolume indicator (2-obj)")
    _plot_region_analysis(exp_dir, METHODS, "exp1_regions.png")


# ── Exp 2 ─────────────────────────────────────────────────────────────────────

def plot_exp2():
    """
    Exp 2 — Three-objective scalability.

    (A) Training curves  (B–D) Three pairwise 2-D Pareto projections.
    Separate figure: exp2_hypervolume.png
    """
    exp_dir    = os.path.join(RESULTS_DIR, "exp2")
    proj_pairs = [
        (0, 1, "Treasure value",  "Time penalty"),
        (0, 2, "Treasure value",  "Fuel cost"),
        (1, 2, "Time penalty",    "Fuel cost"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)

    for mname, minfo in METHODS.items():
        mdir = os.path.join(exp_dir, mname)
        if not os.path.exists(mdir):
            continue
        d = np.load(os.path.join(mdir, "summary.npz"))

        # Panel A: training curves
        smoothed = np.array([_smooth(r) for r in d["train_returns"]])
        mean = smoothed.mean(axis=0)
        std  = smoothed.std(axis=0, ddof=1)
        axes[0].plot(np.arange(len(mean)), mean,
                     color=minfo["color"], label=minfo["label"], linewidth=1.6)
        axes[0].fill_between(np.arange(len(mean)), mean - std, mean + std,
                             alpha=0.15, color=minfo["color"])

        # Panels B–D: Pareto projections
        vecs_list = [np.load(os.path.join(mdir, sf))
                     for sf in sorted(os.listdir(mdir)) if "eval_vecs" in sf]
        if vecs_list:
            vecs = np.mean(vecs_list, axis=0)   # (n_betas, 3)
            for ax_idx, (i, j, xl, yl) in enumerate(proj_pairs):
                axes[ax_idx + 1].scatter(
                    vecs[:, i], vecs[:, j],
                    color=minfo["color"], label=minfo["label"],
                    s=35, alpha=0.88, marker=minfo["marker"],
                    edgecolors="white", linewidths=0.4, zorder=3,
                )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Scalarised return")
    axes[0].set_title("(A) Training curves (3-obj)")
    axes[0].legend(loc="lower right", fontsize=7)

    for ax_idx, (_, _, xl, yl) in enumerate(proj_pairs):
        ax = axes[ax_idx + 1]
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"({chr(66 + ax_idx)}) {xl} vs {yl}")
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle("Exp 2 — Three-objective scalability", fontsize=12)
    _savefig("exp2_results.png")

    _plot_hv_bars(exp_dir, METHODS, "exp2_hypervolume.png",
                  title="Exp 2 — Hypervolume indicator (3-obj)")


# ── Exp 3 ─────────────────────────────────────────────────────────────────────

def plot_exp3():
    """
    Exp 3 — Generalisation across the preference simplex.

    (A) Generalisation gap per method, mean ± 95 % CI (lower = better).
    (B) Seen vs unseen mean scalarised return, illustrating the size of the gap.
    """
    exp_dir = os.path.join(RESULTS_DIR, "exp3")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    bar_x, bar_means, bar_cis, bar_colors = [], [], [], []

    for mname, minfo in METHODS.items():
        mdir = os.path.join(exp_dir, mname)
        if not os.path.exists(mdir):
            continue
        d = np.load(os.path.join(mdir, "summary.npz"))

        bar_x.append(minfo["label"])
        bar_means.append(float(d["gap_mean"]))
        bar_cis.append(float(d["gap_ci95"]))
        bar_colors.append(minfo["color"])

        # Panel B: paired seen / unseen bars (two separate calls — alpha must be scalar)
        seen_m   = float(d["seen_means"].mean())
        unseen_m = float(d["unseen_means"].mean())
        axes[1].bar([minfo["label"] + "\n(seen)"],   [seen_m],
                    color=minfo["color"], alpha=1.0,  width=0.4)
        axes[1].bar([minfo["label"] + "\n(unseen)"], [unseen_m],
                    color=minfo["color"], alpha=0.42, width=0.4)

    # Panel A: gap bar chart
    xs = np.arange(len(bar_x))
    bars = axes[0].bar(xs, bar_means, color=bar_colors, width=0.5, alpha=0.88,
                       yerr=bar_cis, capsize=5,
                       error_kw={"linewidth": 1.2, "ecolor": "#374151"})
    axes[0].axhline(0, color="#374151", linewidth=0.7, linestyle="--", alpha=0.6)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(bar_x)
    axes[0].set_ylabel("Generalisation gap  (lower = better)")
    axes[0].set_title("(A) Generalisation gap  (95 % CI)")
    axes[0].grid(axis="y")

    axes[1].set_ylabel("Mean scalarised return")
    axes[1].set_title("(B) Seen vs unseen preference vectors")
    axes[1].grid(axis="y")
    plt.setp(axes[1].get_xticklabels(), fontsize=7.5)

    fig.suptitle("Exp 3 — Generalisation across preference simplex", fontsize=12)
    _savefig("exp3_generalisation.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()

    dispatch = {0: plot_exp0, 1: plot_exp1, 2: plot_exp2, 3: plot_exp3}

    for idx in sorted(args.exp):
        exp_dir = os.path.join(RESULTS_DIR, f"exp{idx}")
        if not os.path.exists(exp_dir):
            print(f"  [skip] results/exp{idx} not found — run the experiment first")
            continue
        print(f"\nPlotting Exp {idx}...")
        dispatch[idx]()


if __name__ == "__main__":
    main()
