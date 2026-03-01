"""
run_simulation.py — animated simulation entry point.

Usage:
    python scripts/run_simulation.py [--steps N] [--seed S] [--interval MS] [--no-anim]

Examples:
    python scripts/run_simulation.py --steps 40
    python scripts/run_simulation.py --steps 80 --seed 123 --interval 200
    python scripts/run_simulation.py --steps 50 --no-anim   # headless, saves stats plot
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from model import MathematicalWorld, Theorem, Mathematician, SimParams
from model.visualization import animate, run_and_collect, plot_snapshot


def build_initial_world(rng: np.random.Generator, params: SimParams) -> MathematicalWorld:
    """
    Create a small initial world with 5 theorems and 4 mathematicians.
    Modify this function to explore different initial conditions.
    """
    world = MathematicalWorld()

    # ---- Theorems ----
    # (id, [x, y], difficulty)
    theorem_specs = [
        ("T1", [0.15, 0.20], 2.0),   # very hard
        ("T2", [0.20, 0.25], 1.0),   # hard
        ("T3", [0.75, 0.70], 0.5),
        ("T4", [0.50, 0.50], 0.0),   # moderate
        ("T5", [0.50, 0.55], -0.5),  # easy
    ]
    for tid, loc, diff in theorem_specs:
        world.add_theorem(Theorem(
            theorem_id=tid,
            location=np.array(loc),
            difficulty=diff,
        ))

    # Ground-truth implication structure (harder → easier; MD-T6)
    world.add_implication("T1", "T2")
    world.add_implication("T2", "T4")
    world.add_implication("T4", "T5")

    # ---- Mathematicians ----
    # (id, [x, y], ability, theorem_radius, peer_radius)
    mathematician_specs = [
        ("Alice", [0.18, 0.22], 1.5,  0.25, 0.40),  # expert, near cluster 1
        ("Bob",   [0.75, 0.68], 0.3,  0.20, 0.35),  # weaker, near T3
        ("Carol", [0.50, 0.50], 0.8,  0.45, 0.55),  # generalist
        ("Dave",  [0.20, 0.30], 0.0,  0.30, 0.45),  # novice, near T1/T2
    ]
    for mid, loc, ability, t_radius, p_radius in mathematician_specs:
        world.add_mathematician(Mathematician(
            mathematician_id=mid,
            location=np.array(loc),
            ability=ability,
            theorem_radius=t_radius,
            peer_radius=p_radius,
        ))

    return world


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the math importance model simulation.")
    parser.add_argument("--steps",    type=int,   default=40,  help="Number of simulation steps")
    parser.add_argument("--seed",     type=int,   default=42,  help="RNG seed")
    parser.add_argument("--interval", type=int,   default=300, help="Animation frame interval (ms)")
    parser.add_argument("--no-anim",  action="store_true",     help="Skip animation; save stats plot")
    args = parser.parse_args()

    params = SimParams()
    rng = np.random.default_rng(args.seed)
    world = build_initial_world(rng, params)

    print(f"Starting simulation: {args.steps} steps, seed={args.seed}")
    print(world.summary())

    if args.no_anim:
        # Headless run — collect stats and save a final snapshot
        history = run_and_collect(world, params, n_steps=args.steps, seed=args.seed)
        fig = plot_snapshot(world, step_num=args.steps, stats_history=history)
        out = "simulation_snapshot.png"
        fig.savefig(out, dpi=150)
        print(f"\nSnapshot saved to {out}")
        print(world.summary())
    else:
        # Animated run
        anim = animate(world, params, n_steps=args.steps,
                       interval_ms=args.interval, seed=args.seed)
        plt.show()


if __name__ == "__main__":
    main()
