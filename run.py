"""
Entry point for running experiments.

Usage:
    python run.py              # run all experiments in order
    python run.py --exp 1      # run only Exp 1
    python run.py --exp 1 2    # run Exp 1 and Exp 2
    python run.py --seeds 42 123  # override seeds
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   nargs="+", type=int, default=[0, 1, 2, 3],
                        help="Which experiments to run (0 1 2 3)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seeds (default: 42 123 256 512 1024)")
    return parser.parse_args()


def main():
    args  = parse_args()
    seeds = args.seeds or [42, 123, 256, 512, 1024]

    exp_map = {
        0: ("experiments.exp0_single_obj",   "Exp 0 — Single-obj sanity check"),
        1: ("experiments.exp1_two_obj",      "Exp 1 — Two-objective benchmark"),
        2: ("experiments.exp2_three_obj",    "Exp 2 — Three-objective scalability"),
        3: ("experiments.exp3_generalisation","Exp 3 — Generalisation"),
    }

    for idx in sorted(args.exp):
        module_path, label = exp_map[idx]
        print(f"\n{'='*60}")
        print(f"  Running {label}")
        print(f"  Seeds: {seeds}")
        print(f"{'='*60}")
        import importlib
        mod = importlib.import_module(module_path)
        mod.run(seeds=seeds)

    print("\nAll requested experiments complete.")


if __name__ == "__main__":
    main()
