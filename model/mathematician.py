"""
Mathematician class for the mathematical world model.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-M1: ABILITY IS IN [0,1]
#   Ability is a ground-truth real number in [0,1].
#   0 = no mathematical ability, 1 = perfect / omniscient.
#   Rationale: Bounded domain allows Beta-distribution beliefs by peers.
#   Alternative: unbounded (e.g. log-normal), IQ-style scale.
#
# MD-M2: TWO SEPARATE RADII
#   theorem_radius: perception radius for theorems.
#   peer_radius:    perception radius for other mathematicians.
#   These may differ (e.g., a mathematician might track many theorems but
#   only closely follow a few peers). They are separate constants that must
#   be set explicitly (no default) to force conscious model choices.
#   Alternative: one shared radius for all entity types.
#
# MD-M3: HARD BOUNDARY (STEP FUNCTION)
#   Mathematicians have NO beliefs about entities strictly outside their radius.
#   The transition is sharp, not a soft decay with distance.
#   Alternative: Gaussian kernel decay so that distant entities are believed
#   less confidently rather than not at all.
#
# MD-M4: THREE BELIEF TYPES (INDEPENDENT)
#   For each theorem within theorem_radius:
#     (a) Beta belief over its importance  [0,1]
#     (b) Beta belief over its difficulty  [0,1]
#   For each peer within peer_radius:
#     (c) Beta belief over their ability   [0,1]
#   These three belief types are treated as independent.
#   Alternative: joint distribution (e.g., a bivariate Beta or copula) to
#   capture correlation between importance and difficulty.
#
# MD-M5: UNINFORMATIVE INITIAL PRIOR
#   On first encountering any entity, beliefs are initialised to Beta(1,1)
#   (uniform). The mathematician literally "knows nothing yet."
#   Alternative: an informative community-wide prior based on global statistics.
# ============================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from .beliefs import BetaBelief


@dataclass
class TheoremBeliefs:
    """
    A mathematician's beliefs about a single theorem's two latent parameters.

    importance: Beta belief over the theorem's importance ∈ [0,1].
    difficulty: Beta belief over the theorem's difficulty ∈ [0,1].

    MD-M4: Two independent Beta beliefs per theorem.
    MD-M5: Both initialised to Beta(1,1) (uninformative).
    """

    importance: BetaBelief = field(default_factory=BetaBelief)
    difficulty: BetaBelief = field(default_factory=BetaBelief)

    def __repr__(self) -> str:
        return f"TheoremBeliefs(importance={self.importance}, difficulty={self.difficulty})"


@dataclass
class Mathematician:
    """
    Represents a mathematician in 2D mathematical space.

    Attributes:
        mathematician_id: Unique string identifier.
        location:         2D position in [0,1]×[0,1]. See Theorem MD-T1.
        ability:          Ground-truth ability in [0,1]. See MD-M1.
        theorem_radius:   Euclidean radius for theorem perception. See MD-M2, MD-M3.
        peer_radius:      Euclidean radius for peer perception. See MD-M2, MD-M3.
        theorem_beliefs:  theorem_id → TheoremBeliefs. Only for theorems within theorem_radius.
        peer_beliefs:     mathematician_id → BetaBelief (over ability). Only within peer_radius.
    """

    mathematician_id: str
    location: np.ndarray    # shape (2,)
    ability: float          # ∈ [0,1]; MD-M1
    theorem_radius: float   # MD-M2: explicit, no default (see module docstring)
    peer_radius: float      # MD-M2: explicit, no default

    # Populated automatically by World when entities are added
    theorem_beliefs: Dict[str, TheoremBeliefs] = field(default_factory=dict)
    peer_beliefs: Dict[str, BetaBelief] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.location = np.asarray(self.location, dtype=float)
        if self.location.shape != (2,):
            raise ValueError("location must be a 2-element array.")
        if not (0.0 <= self.ability <= 1.0):
            raise ValueError(f"ability must be in [0,1], got {self.ability}")
        if self.theorem_radius < 0:
            raise ValueError("theorem_radius must be non-negative.")
        if self.peer_radius < 0:
            raise ValueError("peer_radius must be non-negative.")

    # ------------------------------------------------------------------
    # Perception predicates (MD-M3: hard boundary)
    # ------------------------------------------------------------------

    def distance_to(self, other_location: np.ndarray) -> float:
        """Euclidean distance to another point. See Theorem MD-T2."""
        return float(np.linalg.norm(self.location - np.asarray(other_location, dtype=float)))

    def can_perceive_theorem(self, theorem_location: np.ndarray) -> bool:
        """True if the theorem is within theorem_radius. MD-M3: hard cutoff."""
        return self.distance_to(theorem_location) <= self.theorem_radius

    def can_perceive_peer(self, peer_location: np.ndarray) -> bool:
        """True if the peer is within peer_radius. MD-M3: hard cutoff."""
        return self.distance_to(peer_location) <= self.peer_radius

    # ------------------------------------------------------------------
    # Belief summaries
    # ------------------------------------------------------------------

    def belief_summary(self) -> dict:
        """Return posterior means for all current beliefs (for inspection/logging)."""
        return {
            "theorem_beliefs": {
                tid: {
                    "importance_mean": tb.importance.mean(),
                    "difficulty_mean": tb.difficulty.mean(),
                }
                for tid, tb in self.theorem_beliefs.items()
            },
            "peer_beliefs": {
                pid: pb.mean()
                for pid, pb in self.peer_beliefs.items()
            },
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Mathematician(id={self.mathematician_id!r}, "
            f"loc=({self.location[0]:.2f}, {self.location[1]:.2f}), "
            f"ability={self.ability:.2f}, "
            f"t_radius={self.theorem_radius:.2f}, p_radius={self.peer_radius:.2f}, "
            f"knows {len(self.theorem_beliefs)} theorems, {len(self.peer_beliefs)} peers)"
        )
