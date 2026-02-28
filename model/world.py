"""
MathematicalWorld: the simulation container.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-W1: SPACE IS [0,1]×[0,1]
#   The world is the unit square. All locations must lie within it.
#   Alternative: torus (to avoid boundary effects), higher-dimensional space.
#
# MD-W2: BELIEF INITIALISATION TIMING
#   When an entity is added to the world, all relevant beliefs are
#   initialised immediately (not lazily at first interaction).
#   Rationale: simpler; matches the idea that mathematicians are
#   continuously aware of everything in their radius.
#   Alternative: beliefs initialised only at the first time step in which
#   the mathematician "actively looks around."
#
# MD-W3: MUTUAL AWARENESS ON MATHEMATICIAN ADDITION
#   When a new mathematician M is added, any existing mathematician P
#   who is within P's peer_radius of M will gain a belief entry for M.
#   Awareness is not necessarily symmetric (P's peer_radius may ≠ M's).
#   Alternative: awareness requires both parties to be within each other's
#   radius (symmetric / mutual).
#
# MD-W4: IMPLICATION IS A DIRECTED RELATION
#   add_implication(A, B) means "A implies B" (A → B in the DAG).
#   No cycle detection is enforced; cycles would be logically inconsistent
#   but the model does not prevent them.
#   Alternative: enforce DAG by topological sort; weight implications by strength.
# ============================================================
"""

import numpy as np
from typing import Dict, List

from .theorem import Theorem
from .mathematician import Mathematician, TheoremBeliefs
from .beliefs import BetaBelief


class MathematicalWorld:
    """
    Container and coordinator for all mathematicians and theorems.

    Responsibilities:
      - Store and index entities by ID.
      - Initialise Bayesian beliefs on entity arrival (MD-W2, MD-W3).
      - Manage the implication graph (MD-W4).
      - Provide neighbourhood queries.
    """

    def __init__(self) -> None:
        self.theorems: Dict[str, Theorem] = {}
        self.mathematicians: Dict[str, Mathematician] = {}

    # ------------------------------------------------------------------
    # Adding entities
    # ------------------------------------------------------------------

    def add_theorem(self, theorem: Theorem) -> None:
        """
        Add a theorem to the world.

        After insertion, any existing mathematician whose theorem_radius
        covers the new theorem gets an uninformative prior over it. (MD-W2)

        Raises:
            ValueError: if theorem_id is already registered.
        """
        if theorem.theorem_id in self.theorems:
            raise ValueError(f"Theorem '{theorem.theorem_id}' already exists.")
        self._validate_location(theorem.location, label=f"Theorem {theorem.theorem_id}")
        self.theorems[theorem.theorem_id] = theorem

        # MD-W2: initialise beliefs for all existing mathematicians in range
        for m in self.mathematicians.values():
            if m.can_perceive_theorem(theorem.location):
                if theorem.theorem_id not in m.theorem_beliefs:
                    m.theorem_beliefs[theorem.theorem_id] = TheoremBeliefs()  # MD-M5

    def add_mathematician(self, mathematician: Mathematician) -> None:
        """
        Add a mathematician to the world.

        Initialises:
          - Their theorem_beliefs for all theorems within theorem_radius.
          - Their peer_beliefs for all existing mathematicians within peer_radius.
          - peer_beliefs of existing mathematicians who can perceive the newcomer. (MD-W3)

        Raises:
            ValueError: if mathematician_id is already registered.
        """
        if mathematician.mathematician_id in self.mathematicians:
            raise ValueError(f"Mathematician '{mathematician.mathematician_id}' already exists.")
        self._validate_location(mathematician.location, label=f"Mathematician {mathematician.mathematician_id}")

        self.mathematicians[mathematician.mathematician_id] = mathematician

        # MD-W2: initialise newcomer's theorem beliefs
        for theorem in self.theorems.values():
            if mathematician.can_perceive_theorem(theorem.location):
                mathematician.theorem_beliefs[theorem.theorem_id] = TheoremBeliefs()  # MD-M5

        # MD-W2 + MD-W3: initialise newcomer's peer beliefs, and update peers
        for peer in self.mathematicians.values():
            if peer.mathematician_id == mathematician.mathematician_id:
                continue

            # Newcomer's belief about this peer
            if mathematician.can_perceive_peer(peer.location):
                mathematician.peer_beliefs[peer.mathematician_id] = BetaBelief()  # MD-M5

            # Peer's belief about the newcomer (MD-W3: not necessarily symmetric)
            if peer.can_perceive_peer(mathematician.location):
                if mathematician.mathematician_id not in peer.peer_beliefs:
                    peer.peer_beliefs[mathematician.mathematician_id] = BetaBelief()  # MD-M5

    # ------------------------------------------------------------------
    # Implication graph
    # ------------------------------------------------------------------

    def add_implication(self, source_id: str, target_id: str) -> None:
        """
        Record that the theorem `source_id` directly implies `target_id`.

        Updates both the source's `implies` list and the target's `implied_by`
        list for fast bidirectional traversal. (MD-W4)

        Raises:
            ValueError: if either theorem ID is not registered.
        """
        if source_id not in self.theorems:
            raise ValueError(f"Source theorem '{source_id}' not found.")
        if target_id not in self.theorems:
            raise ValueError(f"Target theorem '{target_id}' not found.")
        self.theorems[source_id]._add_implies(target_id)
        self.theorems[target_id]._add_implied_by(source_id)

    # ------------------------------------------------------------------
    # Neighbourhood queries
    # ------------------------------------------------------------------

    def theorems_near(self, location: np.ndarray, radius: float) -> List[Theorem]:
        """Return all theorems within Euclidean `radius` of `location`."""
        loc = np.asarray(location, dtype=float)
        return [t for t in self.theorems.values() if t.distance_to(loc) <= radius]

    def mathematicians_near(self, location: np.ndarray, radius: float) -> List[Mathematician]:
        """Return all mathematicians within Euclidean `radius` of `location`."""
        loc = np.asarray(location, dtype=float)
        return [m for m in self.mathematicians.values() if m.distance_to(loc) <= radius]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_location(self, location: np.ndarray, label: str = "") -> None:
        """
        Check that a location lies within [0,1]×[0,1]. (MD-W1)

        # MD-W1 (constant): Locations outside [0,1]×[0,1] are rejected.
        # If you want a larger or toroidal space, change the bounds here.
        """
        if not (np.all(location >= 0.0) and np.all(location <= 1.0)):
            raise ValueError(
                f"{label} location {location} is outside [0,1]×[0,1]. (MD-W1)"
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable snapshot of the world."""
        lines = [
            f"MathematicalWorld: {len(self.theorems)} theorem(s), "
            f"{len(self.mathematicians)} mathematician(s)",
            "",
            "--- Theorems ---",
        ]
        for t in self.theorems.values():
            lines.append(f"  {t}")

        lines += ["", "--- Mathematicians ---"]
        for m in self.mathematicians.values():
            lines.append(f"  {m}")
            for tid, tb in m.theorem_beliefs.items():
                lines.append(
                    f"      belief[{tid}]: "
                    f"imp={tb.importance.mean():.2f}±{tb.importance.variance():.4f}  "
                    f"diff={tb.difficulty.mean():.2f}±{tb.difficulty.variance():.4f}"
                )
            for pid, pb in m.peer_beliefs.items():
                lines.append(
                    f"      belief[peer {pid}]: "
                    f"ability={pb.mean():.2f}±{pb.variance():.4f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MathematicalWorld("
            f"theorems={len(self.theorems)}, "
            f"mathematicians={len(self.mathematicians)})"
        )
