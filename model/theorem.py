"""
Theorem class for the mathematical world model.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-T1: MATHEMATICAL SPACE
#   Theorems live in the unit square [0,1]×[0,1].
#   Proximity in this space represents conceptual similarity / topical closeness
#   (e.g., two number-theory theorems would cluster together).
#   Alternative: higher-dimensional space, hyperbolic space (for hierarchical
#   topic trees), or discrete topic categories.
#
# MD-T2: DISTANCE METRIC
#   Euclidean distance in [0,1]×[0,1].
#   Alternative: L1 (Manhattan), hyperbolic distance, cosine similarity.
#
# MD-T3: DIFFICULTY IS IN [0,1]
#   Difficulty is a real number in [0,1]. 0 = trivial, 1 = currently intractable.
#   Rationale: Bounded domain allows Beta-distribution beliefs; normalisation
#   simplifies comparison across theorems.
#   Alternative: unbounded positive real (log-normal prior), ordinal scale.
#
# MD-T4: IMPORTANCE IS IN [0,1] (LATENT GROUND TRUTH)
#   Importance is the hidden quantity that mathematicians try to estimate.
#   It represents "how fundamentally significant" the theorem is to mathematics.
#   Mathematicians hold Beta beliefs over this; it is never directly revealed.
#   Alternative: importance emerges solely from implication structure (PageRank-like).
#
# MD-T5: IMPLICATION IS A DIRECTED GRAPH (ADJACENCY LISTS)
#   Each theorem stores two lists: theorems it implies and theorems that imply it.
#   This is a redundant bidirectional adjacency list for fast lookup in both directions.
#   No cycle detection is enforced, but the intended structure is a DAG.
#   Alternative: a separate NetworkX graph object; edge-weighted implications.
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
        theorem_id:  Unique string identifier (e.g. "Fermat_Last", "T1").
        location:    2D position in [0,1]×[0,1]. See MD-T1, MD-T2.
        difficulty:  Ground-truth difficulty in [0,1]. See MD-T3.
        importance:  Ground-truth importance in [0,1]. Latent; see MD-T4.
        implies:     IDs of theorems directly implied by this one. See MD-T5.
        implied_by:  IDs of theorems that directly imply this one. See MD-T5.
    """

    theorem_id: str
    location: np.ndarray   # shape (2,); MD-T1
    difficulty: float       # ∈ [0,1]; MD-T3
    importance: float       # ∈ [0,1]; MD-T4

    # MD-T5: adjacency lists for the implication DAG
    implies: List[str] = field(default_factory=list)
    implied_by: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.location = np.asarray(self.location, dtype=float)
        if self.location.shape != (2,):
            raise ValueError("location must be a 2-element array.")
        if not (0.0 <= self.difficulty <= 1.0):
            raise ValueError(f"difficulty must be in [0,1], got {self.difficulty}")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(f"importance must be in [0,1], got {self.importance}")

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
            f"diff={self.difficulty:.2f}, imp={self.importance:.2f}, "
            f"implies={self.implies}, implied_by={self.implied_by})"
        )
