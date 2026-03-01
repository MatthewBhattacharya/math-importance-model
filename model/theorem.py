"""
Theorem class for the mathematical world model.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-T1: MATHEMATICAL SPACE
#   Theorems live in the unit square [0,1]×[0,1].
#   Proximity = conceptual/topical similarity.
#   Alternative: higher-dimensional, hyperbolic, discrete topic categories.
#
# MD-T2: DISTANCE METRIC
#   Euclidean distance in [0,1]×[0,1].
#   Alternative: L1, hyperbolic, cosine similarity.
#
# MD-T3: DIFFICULTY IS AN UNBOUNDED REAL
#   Difficulty is any real number. No lower or upper bound.
#   Positive = hard, negative = trivial, 0 = "average".
#   Rationale: allows a Gaussian prior (Normal(0, σ²)) on difficulty and
#   a clean sigmoid success-probability function of (ability − difficulty).
#   Alternative: bounded [0,1] (previous version), log-normal (positive only).
#
# MD-T4: NO GROUND-TRUTH IMPORTANCE
#   Theorems do NOT have an intrinsic importance value.
#   Importance is entirely a collective belief held by mathematicians.
#   The dynamics model how these beliefs form and propagate.
#   Alternative: derive importance from the implication graph (PageRank-like).
#
# MD-T5: IMPLICATION IS A DIRECTED GRAPH (ADJACENCY LISTS)
#   Each theorem stores two lists: theorems it implies and theorems that imply it.
#   Bidirectional adjacency for fast traversal in both directions.
#   No cycle detection; intended structure is a DAG.
#   Alternative: separate NetworkX graph object; weighted edges.
#
# MD-T6: DIFFICULTY MONOTONICITY IN IMPLICATION
#   When creating implications (especially during spawning), a theorem
#   of lower difficulty is never allowed to imply one of higher difficulty.
#   Rationale: models the intuition that powerful/harder results imply
#   simpler corollaries. Enforced by the spawning logic in dynamics.py.
#   Alternative: allow any direction; use difficulty difference as a weight.
# ============================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Theorem:
    """
    Represents a mathematical theorem in 2D mathematical space.

    Attributes:
        theorem_id:  Unique string identifier.
        location:    2D position in [0,1]×[0,1]. See MD-T1, MD-T2.
        difficulty:  Ground-truth difficulty (unbounded real). See MD-T3.
                     Unknown to mathematicians; they form NormalBelief about it.
        implies:     IDs of theorems directly implied by this one. See MD-T5, MD-T6.
        implied_by:  IDs of theorems that directly imply this one. See MD-T5.

    Note: there is NO importance field. See MD-T4.
    """

    theorem_id: str
    location: np.ndarray   # shape (2,), values in [0,1]×[0,1]
    difficulty: float       # unbounded real; MD-T3

    # MD-T5: directed implication adjacency lists
    implies: List[str] = field(default_factory=list)
    implied_by: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.location = np.asarray(self.location, dtype=float)
        if self.location.shape != (2,):
            raise ValueError("location must be a 2-element array.")
        # MD-T3: difficulty is unbounded — no range check performed.
        # MD-T1: locations must lie in [0,1]×[0,1].
        if not (np.all(self.location >= 0.0) and np.all(self.location <= 1.0)):
            raise ValueError(f"location {self.location} is outside [0,1]×[0,1].")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def distance_to(self, other_location: np.ndarray) -> float:
        """
        Euclidean distance to another point in mathematical space.
        MD-T2: Euclidean metric on [0,1]×[0,1].
        """
        return float(np.linalg.norm(self.location - np.asarray(other_location, dtype=float)))

    # ------------------------------------------------------------------
    # Implication graph helpers (called by World to keep lists consistent)
    # ------------------------------------------------------------------

    def _add_implies(self, target_id: str) -> None:
        if target_id not in self.implies:
            self.implies.append(target_id)

    def _add_implied_by(self, source_id: str) -> None:
        if source_id not in self.implied_by:
            self.implied_by.append(source_id)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Theorem(id={self.theorem_id!r}, "
            f"loc=({self.location[0]:.2f}, {self.location[1]:.2f}), "
            f"diff={self.difficulty:.2f}, "
            f"implies={self.implies}, implied_by={self.implied_by})"
        )
