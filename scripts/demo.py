"""
demo.py — minimal demonstration of the skeleton.

Run from the project root:
    python scripts/demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from model import MathematicalWorld, Theorem, Mathematician

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# Build a tiny world
# -----------------------------------------------------------------------

world = MathematicalWorld()

# Four theorems scattered in [0,1]×[0,1]
# (theorem_id, location, difficulty, importance)
theorem_specs = [
    ("T1", [0.20, 0.30], 0.30, 0.90),  # easy, very important
    ("T2", [0.80, 0.70], 0.70, 0.50),  # hard, moderately important
    ("T3", [0.25, 0.35], 0.50, 0.60),  # near T1
    ("T4", [0.50, 0.50], 0.40, 0.40),  # central
]

for tid, loc, diff, imp in theorem_specs:
    world.add_theorem(Theorem(
        theorem_id=tid,
        location=np.array(loc),
        difficulty=diff,
        importance=imp,
    ))

# T1 → T3 → T4 (implication chain)
world.add_implication("T1", "T3")
world.add_implication("T3", "T4")

# Three mathematicians
# (id, location, ability, theorem_radius, peer_radius)
#
# MODELING DECISION (constant): radii chosen to illustrate partial overlap.
# Alice sits near T1/T3 (theorem_radius=0.30 covers them but not T2/T4).
# Bob sits near T2 (theorem_radius=0.20 covers only T2).
# Carol is central (theorem_radius=0.50 covers everything).
mathematician_specs = [
    ("Alice", [0.20, 0.30], 0.80, 0.30, 0.50),
    ("Bob",   [0.80, 0.70], 0.50, 0.20, 0.40),
    ("Carol", [0.50, 0.50], 0.65, 0.50, 0.60),
]

for mid, loc, ability, t_radius, p_radius in mathematician_specs:
    world.add_mathematician(Mathematician(
        mathematician_id=mid,
        location=np.array(loc),
        ability=ability,
        theorem_radius=t_radius,
        peer_radius=p_radius,
    ))

# -----------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------

print(world.summary())
print()

# Show that Alice's beliefs are initialised
alice = world.mathematicians["Alice"]
print(f"Alice's theorem beliefs (prior means = 0.5 everywhere — MD-M5):")
for tid, tb in alice.theorem_beliefs.items():
    print(f"  {tid}: importance {tb.importance}  difficulty {tb.difficulty}")

print(f"\nAlice's peer beliefs:")
for pid, pb in alice.peer_beliefs.items():
    print(f"  {pid}: ability {pb}")

# Simulate one noisy observation: Alice sees that T1 is easy (difficulty ≈ 0.25)
# and updates her belief about T1's difficulty.
print("\n--- Alice observes T1 has difficulty ~0.25 (noisy signal) ---")
alice.theorem_beliefs["T1"].difficulty.update(observation=0.25, weight=1.0)
print(f"  T1 difficulty belief after update: {alice.theorem_beliefs['T1'].difficulty}")
