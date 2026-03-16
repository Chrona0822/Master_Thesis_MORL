# Multi-Objective Reinforcement Learning — Master's Thesis

Empirical comparison of preference-conditioned agents on the **Deep Sea Treasure** benchmark, covering two- and three-objective settings.

---

## Methods

| Method            | Type              | Description                                                                                                             |
| ----------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Cond-DQN**      | Neural (proposed) | Preference-conditioned DQN; β concatenated to state input; trained with experience replay                               |
| **Tabular + GIP** | Tabular baseline  | Maintains one Q-table per simplex grid point; updates all tables simultaneously (Generalised Policy Improvement)        |
| **Envelope Q**    | Tabular baseline  | Envelope Bellman backup (Yang et al., 2019); bootstraps across Q-tables so information flows between preference vectors |

---

## Experiments

| ID    | Description                                     | Key metric                            |
| ----- | ----------------------------------------------- | ------------------------------------- |
| Exp 0 | DQN convergence check (fixed β = [0.5, 0.5])    | Training curve                        |
| Exp 1 | Two-objective benchmark (21 evaluation betas)   | Hypervolume, scalarised return vs β   |
| Exp 2 | Three-objective scalability (fuel-cost wrapper) | Hypervolume, Pareto projections       |
| Exp 3 | Generalisation across preference simplex        | Generalisation gap (seen vs unseen β) |
| Exp 4 | Sensitivity analysis (lr / grid size sweep)     | Hypervolume per hyperparameter config |

---

## Project Structure

```
├── agents/
│   ├── dqn_agent.py          # Preference-conditioned DQN
│   ├── tabular_agent.py      # Tabular + GIP
│   ├── envelope_agent.py     # Envelope Q-learning
│   └── pareto_agent.py       # Pareto Q-learning (reference implementation)
├── envs/
│   └── dst_fuel.py           # Three-objective DST wrapper (adds fuel cost)
├── experiments/
│   ├── exp0_single_obj.py
│   ├── exp1_two_obj.py
│   ├── exp2_three_obj.py
│   ├── exp3_generalisation.py
│   ├── exp4_sensitivity.py
│   └── train_utils.py        # Shared training loops
├── evaluation/
│   └── metrics.py            # Hypervolume, generalisation gap, evaluation helpers
├── results/                  # Auto-generated experiment outputs (git-ignored)
├── run.py                    # Unified entry point
└── plot_result_new.py        # Plotting script (Cond-DQN / Tabular / Envelope Q)
```

---

## Setup

```bash
pip install mo-gymnasium torch numpy scipy matplotlib
```

Tested with Python 3.10+.

---

## Running Experiments

```bash
# Run all experiments (all methods, 5 seeds)
python run.py

# Run a specific experiment
python run.py --exp 1

# Run multiple experiments
python run.py --exp 1 2

# Run a specific method only
python run.py --exp 1 --method dqn
python run.py --exp 1 --method tabular envelope

# Override seeds
python run.py --exp 1 --seeds 42 123
```

Available experiments: `0 1 2 3 4`
Available methods: `dqn tabular pareto envelope`

---

## Plotting Results

```bash
# Plot all experiments
python plot_result_new.py

# Plot specific experiments
python plot_result_new.py --exp 1
python plot_result_new.py --exp 0 1 2 4
```

Figures are saved to `results/figures/<timestamp>/`.

---

## Environment

**Deep Sea Treasure (concave variant)** from [mo-gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

- 2-objective: treasure value + time penalty
- 3-objective: treasure value + time penalty + fuel cost (custom wrapper)
- State: (row, col) grid position
- Actions: up / down / left / right

---
