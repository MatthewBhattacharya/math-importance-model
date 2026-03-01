"""
MathematicalWorld — simulation container and coordinator.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-W1: SPACE IS [0,1]×[0,1]
#   All entity locations must lie in the unit square.
#   Alternative: torus (no boundary effects), larger bounding box.
#
# MD-W2: BELIEF INITIALISATION TIMING
#   Beliefs are initialised immediately when an entity is added.
#   Alternative: lazy initialisation at the first time step.
#
# MD-W3: ASYMMETRIC PEER AWARENESS
#   When new mathematician M is added, existing peer P gains a belief
#   about M if M ∈ P's peer_radius — regardless of M's own peer_radius.
#   Alternative: symmetric (both must be within each other's radius).
#
# MD-W4: IMPLICATION IS DIRECTED; NO CYCLE DETECTION
#   add_implication(A, B) records A → B. Cycles are not prevented.
#   Alternative: enforce DAG by topological sort.
#
# MD-W5: COMMUNITY KNOWLEDGE TRACKING
#   world.known_implications tracks every implication discovered by ANY
#   mathematician. This is separate from individual mathematician knowledge.
# ============================================================
"""

import numpy as np
from typing import Dict, List, Set, Tuple

from .theorem import Theorem
from .mathematician import Mathematician, TheoremBeliefs
from .beliefs import NormalBelief


class MathematicalWorld:
    """
    Container and coordinator for all mathematicians and theorems.

    Responsibilities:
      - Store and index entities by ID.
      - Initialise Bayesian beliefs on entity arrival (MD-W2, MD-W3).
      - Manage the implication graph (MD-W4).
      - Track community-level discovered implications (MD-W5).
      - Provide neighbourhood queries.
    """

    def __init__(self) -> None:
        self.theorems: Dict[str, Theorem] = {}
        self.mathematicians: Dict[str, Mathematician] = {}

        # MD-W5: community knowledge — any implication discovered by at least one mathematician
        self.known_implications: Set[Tuple[str, str]] = set()

        # Monotonically increasing counters for auto-naming spawned entities
        self._theorem_counter: int = 0
        self._mathematician_counter: int = 0

    # ------------------------------------------------------------------
    # Adding entities
    # ------------------------------------------------------------------

    def add_theorem(self, theorem: Theorem) -> None:
        """
        Add a theorem to the world.

        Initialises uninformative beliefs in all existing mathematicians who
        can perceive the new theorem (MD-W2).

        Raises:
            ValueError: if theorem_id is already registered or location out of bounds.
        """
        if theorem.theorem_id in self.theorems:
            raise ValueError(f"Theorem '{theorem.theorem_id}' already exists.")
        self._validate_location(theorem.location, label=f"Theorem {theorem.theorem_id}")
        self.theorems[theorem.theorem_id] = theorem

        # MD-W2: initialise beliefs for existing mathematicians in range
        for m in self.mathematicians.values():
            if m.can_perceive_theorem(theorem.location):
                if theorem.theorem_id not in m.theorem_beliefs:
                    m.theorem_beliefs[theorem.theorem_id] = TheoremBeliefs()  # MD-M5

    def add_mathematician(self, mathematician: Mathematician) -> None:
        """
        Add a mathematician to the world.

        Initialises their beliefs over all existing entities within their radii,
        and updates existing mathematicians' peer beliefs if appropriate (MD-W2, MD-W3).

        Raises:
            ValueError: if mathematician_id is already registered or location out of bounds.
        """
        if mathematician.mathematician_id in self.mathematicians:
            raise ValueError(f"Mathematician '{mathematician.mathematician_id}' already exists.")
        self._validate_location(mathematician.location, label=f"Mathematician {mathematician.mathematician_id}")
        self.mathematicians[mathematician.mathematician_id] = mathematician

        # MD-W2: initialise newcomer's theorem beliefs
        for theorem in self.theorems.values():
            if mathematician.can_perceive_theorem(theorem.location):
                mathematician.theorem_beliefs[theorem.theorem_id] = TheoremBeliefs()  # MD-M5

        # MD-W2 + MD-W3: peer belief initialisation (asymmetric)
        for peer in self.mathematicians.values():
            if peer.mathematician_id == mathematician.mathematician_id:
                continue

            # Newcomer's belief about this peer
            if mathematician.can_perceive_peer(peer.location):
                mathematician.peer_beliefs[peer.mathematician_id] = NormalBelief()  # MD-M5

            # Peer's belief about the newcomer (MD-W3: not necessarily symmetric)
            if peer.can_perceive_peer(mathematician.location):
                if mathematician.mathematician_id not in peer.peer_beliefs:
                    peer.peer_beliefs[mathematician.mathematician_id] = NormalBelief()  # MD-M5

    # ------------------------------------------------------------------
    # Implication graph
    # ------------------------------------------------------------------

    def add_implication(self, source_id: str, target_id: str) -> None:
        """
        Record the ground-truth implication source_id → target_id.

        Updates both theorem adjacency lists (MD-W4).
        Does NOT mark this as discovered (that happens via dynamics).

        Raises:
            ValueError: if either theorem ID is not registered.
        """
        if source_id not in self.theorems:
            raise ValueError(f"Source theorem '{source_id}' not found.")
        if target_id not in self.theorems:
            raise ValueError(f"Target theorem '{target_id}' not found.")
        self.theorems[source_id]._add_implies(target_id)
        self.theorems[target_id]._add_implied_by(source_id)

    def record_discovery(self, mathematician_id: str, source_id: str, target_id: str) -> None:
        """
        Record that mathematician `mathematician_id` discovered source_id → target_id.

        Updates the mathematician's known_implications and the community set (MD-W5).
        Does NOT add it to the ground-truth theorem adjacency lists
        (those are set by add_implication / spawning only).
        """
        m = self.mathematicians[mathematician_id]
        m.known_implications.add((source_id, target_id))
        self.known_implications.add((source_id, target_id))

    # ------------------------------------------------------------------
    # Auto-naming helpers for spawned entities
    # ------------------------------------------------------------------

    def next_theorem_id(self) -> str:
        """Return a unique theorem ID for a spawned theorem (prefix 'Ts' avoids collisions with manually-named theorems)."""
        self._theorem_counter += 1
        tid = f"Ts{self._theorem_counter}"
        # Guarantee uniqueness even if caller uses the same prefix
        while tid in self.theorems:
            self._theorem_counter += 1
            tid = f"Ts{self._theorem_counter}"
        return tid

    def next_mathematician_id(self) -> str:
        """Return a unique mathematician ID for a spawned mathematician (prefix 'Ms')."""
        self._mathematician_counter += 1
        mid = f"Ms{self._mathematician_counter}"
        while mid in self.mathematicians:
            self._mathematician_counter += 1
            mid = f"Ms{self._mathematician_counter}"
        return mid

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

    def nearest_mathematicians(self, mathematician_id: str, n: int) -> List[Mathematician]:
        """
        Return the n closest OTHER mathematicians to the given one, by Euclidean distance.
        Used to find communication neighbours (MD-C16 in params.py).
        """
        m = self.mathematicians[mathematician_id]
        others = [p for p in self.mathematicians.values() if p.mathematician_id != mathematician_id]
        others.sort(key=lambda p: m.distance_to(p.location))
        return others[:n]

    # ------------------------------------------------------------------
    # Aggregate statistics (for visualisation / logging)
    # ------------------------------------------------------------------

    def community_importance(self, theorem_id: str) -> float:
        """
        Mean importance belief across all mathematicians who know this theorem.

        Returns 0.0 if no mathematician knows it.
        """
        values = [
            m.theorem_beliefs[theorem_id].importance.mu
            for m in self.mathematicians.values()
            if theorem_id in m.theorem_beliefs
        ]
        return float(np.mean(values)) if values else 0.0

    def community_difficulty(self, theorem_id: str) -> float:
        """Mean difficulty belief across all mathematicians who know this theorem."""
        values = [
            m.theorem_beliefs[theorem_id].difficulty.mu
            for m in self.mathematicians.values()
            if theorem_id in m.theorem_beliefs
        ]
        return float(np.mean(values)) if values else 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_location(self, location: np.ndarray, label: str = "") -> None:
        """
        Reject locations outside [0,1]×[0,1]. MD-W1.
        To change the space, modify the bounds here.
        """
        if not (np.all(location >= 0.0) and np.all(location <= 1.0)):
            raise ValueError(f"{label} location {location} is outside [0,1]×[0,1]. (MD-W1)")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable snapshot of the world state."""
        lines = [
            f"MathematicalWorld: {len(self.theorems)} theorems, "
            f"{len(self.mathematicians)} mathematicians, "
            f"{len(self.known_implications)} known implications",
            "",
            "--- Theorems ---",
        ]
        for t in self.theorems.values():
            lines.append(
                f"  {t}  community_imp={self.community_importance(t.theorem_id):.2f}"
            )
        lines += ["", "--- Mathematicians ---"]
        for m in self.mathematicians.values():
            lines.append(f"  {m}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MathematicalWorld("
            f"theorems={len(self.theorems)}, "
            f"mathematicians={len(self.mathematicians)}, "
            f"known_implications={len(self.known_implications)})"
        )
