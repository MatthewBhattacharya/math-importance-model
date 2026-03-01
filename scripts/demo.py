"""
demo.py — quick headless check that the dynamics run correctly.

Run from the project root:
    python scripts/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from model import MathematicalWorld, Theorem, Mathematician, SimParams
from model.visualization import run_and_collect

rng = np.random.default_rng(42)
params = SimParams()

# ---- Build a small initial world ----
world = MathematicalWorld()

theorem_specs = [
    ("T1", [0.20, 0.30], 1.5),   # hard
    ("T2", [0.80, 0.70], 0.5),
    ("T3", [0.25, 0.35], 0.0),   # moderate
    ("T4", [0.50, 0.50], -0.5),  # easy
]
for tid, loc, diff in theorem_specs:
    world.add_theorem(Theorem(
        theorem_id=tid,
        location=np.array(loc),
        difficulty=diff,
    ))

# Ground truth: T1 → T3 → T4 (harder implies easier)
world.add_implication("T1", "T3")
world.add_implication("T3", "T4")

mathematician_specs = [
    ("Alice", [0.20, 0.30], 1.0, 0.30, 0.50),   # high ability, near T1/T3
    ("Bob",   [0.80, 0.70], 0.0, 0.20, 0.40),   # average ability, near T2
    ("Carol", [0.50, 0.50], 0.5, 0.50, 0.60),   # central, sees all
]
for mid, loc, ability, t_radius, p_radius in mathematician_specs:
    world.add_mathematician(Mathematician(
        mathematician_id=mid,
        location=np.array(loc),
        ability=ability,
        theorem_radius=t_radius,
        peer_radius=p_radius,
    ))

print("=== Initial state ===")
print(world.summary())
print()

# ---- Run 5 steps ----
history = run_and_collect(world, params, n_steps=5, seed=7)

print("=== After 5 steps ===")
print(world.summary())
print()

print("Statistics per step:")
for i in range(5):
    print(
        f"  step {i+1}: "
        f"mean_importance={history['mean_importance'][i]:.3f}  "
        f"known_impl={history['n_known_impl'][i]}  "
        f"theorems={history['n_theorems'][i]}  "
        f"mathematicians={history['n_mathematicians'][i]}"
    )

print()
print("Alice's beliefs after 5 steps:")
alice = world.mathematicians.get("Alice")
if alice:
    for tid, tb in alice.theorem_beliefs.items():
        print(f"  {tid}: importance={tb.importance}  difficulty={tb.difficulty}")
    print(f"  known implications: {alice.known_implications}")
