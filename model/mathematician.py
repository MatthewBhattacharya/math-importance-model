"""
Mathematician class for the mathematical world model.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-M1: ABILITY IS AN UNBOUNDED REAL
#   Ability is any real number (ground truth). Positive = more capable,
#   negative = below "average". Peers form NormalBelief estimates of it.
#   Alternative: bounded [0,1] (previous version), log-normal (positive only).
#
# MD-M2: TWO SEPARATE RADII
#   theorem_radius: Euclidean radius for perceiving theorems.
#   peer_radius:    Euclidean radius for perceiving other mathematicians.
#   Both must be set explicitly to force a conscious modeling choice.
#   Alternative: one shared radius for all entity types.
#
# MD-M3: HARD STEP-FUNCTION BOUNDARY
#   Mathematicians have no beliefs about entities outside their radius.
#   The transition is sharp (step function), not a soft decay.
#   Alternative: Gaussian kernel — distant entities are believed less
#   confidently rather than not at all.
#
# MD-M4: THREE INDEPENDENT BELIEF TYPES (all NormalBelief)
#   Per theorem within theorem_radius:
#     (a) NormalBelief over theorem's importance  (unbounded real)
#     (b) NormalBelief over theorem's difficulty  (unbounded real)
#   Per peer within peer_radius:
#     (c) NormalBelief over peer's ability        (unbounded real)
#   Treated as independent. Alternative: joint distribution / copula.
#
# MD-M5: UNINFORMATIVE INITIAL PRIOR
#   NormalBelief(mu=0, sigma2=PRIOR_SIGMA2) on first encounter.
#   Alternative: informative prior derived from community statistics.
#
# MD-G1: GOSSIP INFORMATION PACKETS
#   Each mathematician tracks:
#     - self_info: info generated this step (own proof attempts, links found)
#     - gossip_buffer: info RECEIVED last step (to relay this step)
#   When sending to neighbor N, relay self_info + gossip entries whose
#   generator_id != N.mathematician_id (don't relay N's own actions back to N).
#   Alternative: track relay-chain depth; enforce no-duplicate filtering.
#
# MD-G3: RECEIVER-SIDE DEDUPLICATION
#   The same info can arrive via multiple gossip paths (A tells B tells C;
#   A also tells C directly). Each piece of evidence must update beliefs
#   exactly once per receiver to avoid treating the same fact as n independent
#   observations (which would cause overconfident belief compression).
#   - Proof results: Mathematician.processed_proofs tracks (mathematician_id, theorem_id).
#   - Link discoveries: Mathematician.known_implications serves the same role.
#   Alternative: sender-side deduplication (track what each neighbor has seen);
#   this is more complex and still doesn't guard against all multi-path delivery.
# ============================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union

from .beliefs import NormalBelief


# ------------------------------------------------------------------
# Information packet types (for communication / gossip)
# ------------------------------------------------------------------

@dataclass
class ProofAttempt:
    """Records that a mathematician attempted (and succeeded or failed at) a theorem."""
    mathematician_id: str
    theorem_id: str
    success: bool


@dataclass
class LinkDiscovery:
    """
    Records that a mathematician discovered the implication source_id → target_id.

    # MD-G2: Links are directed (source implies target). Discovery is of a
    # pre-existing ground-truth implication that was previously unknown to
    # this mathematician.
    """
    mathematician_id: str
    source_id: str   # source implies target
    target_id: str


@dataclass
class InfoPacket:
    """
    Wraps a piece of information for communication.

    generator_id: ID of the mathematician who GENERATED this info (not the relay).
    Used for gossip filtering: don't send info back to the person who did the thing.
    See MD-G1.
    """
    content: Union[ProofAttempt, LinkDiscovery]
    generator_id: str


# ------------------------------------------------------------------
# Per-theorem belief bundle
# ------------------------------------------------------------------

@dataclass
class TheoremBeliefs:
    """
    A mathematician's beliefs about a single theorem's two latent parameters.

    importance: NormalBelief — how important the theorem is (unbounded).
    difficulty: NormalBelief — how difficult the theorem is (unbounded).

    MD-M4: Two independent NormalBelief distributions per theorem.
    MD-M5: Both initialised to NormalBelief() = N(0, PRIOR_SIGMA2) by default.
    """
    importance: NormalBelief = field(default_factory=NormalBelief)
    difficulty: NormalBelief = field(default_factory=NormalBelief)

    def __repr__(self) -> str:
        return f"TheoremBeliefs(imp={self.importance}, diff={self.difficulty})"


# ------------------------------------------------------------------
# Mathematician
# ------------------------------------------------------------------

@dataclass
class Mathematician:
    """
    Represents a mathematician in 2D mathematical space.

    Attributes:
        mathematician_id: Unique string identifier.
        location:         2D position in [0,1]×[0,1]. See Theorem MD-T1.
        ability:          Ground-truth ability (unbounded real). See MD-M1.
        theorem_radius:   Euclidean radius for theorem perception. See MD-M2, MD-M3.
        peer_radius:      Euclidean radius for peer perception. See MD-M2, MD-M3.
        theorem_beliefs:  theorem_id → TheoremBeliefs. Only within theorem_radius.
        peer_beliefs:     mathematician_id → NormalBelief (ability). Only within peer_radius.
        known_implications: Set of (source_id, target_id) pairs this mathematician
                            has personally discovered.

    Communication fields (managed by dynamics.py, see MD-G1, MD-G3):
        self_info:        InfoPackets generated BY this mathematician THIS step.
        gossip_buffer:    InfoPackets received LAST step (to relay this step).
        processed_proofs: (mathematician_id, theorem_id) pairs already incorporated
                          into beliefs. Guards against double-counting via gossip. MD-G3.
    """

    mathematician_id: str
    location: np.ndarray    # shape (2,)
    ability: float          # unbounded real; MD-M1

    # MD-M2: explicit, no default — force the caller to think about these
    theorem_radius: float
    peer_radius: float

    # Populated by World on entity arrival
    theorem_beliefs: Dict[str, TheoremBeliefs] = field(default_factory=dict)
    peer_beliefs: Dict[str, NormalBelief] = field(default_factory=dict)

    # Implication graph knowledge — personal (a subset of ground truth)
    known_implications: Set[Tuple[str, str]] = field(default_factory=set)

    # MD-G3: DEDUPLICATION — receiver-side evidence tracking
    # Each piece of evidence should update beliefs exactly once per receiver.
    # processed_proofs: set of (mathematician_id, theorem_id) proof results
    #   already incorporated. Prevents the same gossip-relayed proof result
    #   from updating beliefs a second time via a different relay path.
    processed_proofs: Set[Tuple[str, str]] = field(default_factory=set)

    # Communication buffers — managed by dynamics.step()
    self_info: List[InfoPacket] = field(default_factory=list)
    gossip_buffer: List[InfoPacket] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.location = np.asarray(self.location, dtype=float)
        if self.location.shape != (2,):
            raise ValueError("location must be a 2-element array.")
        # MD-M1: ability is unbounded — no range check.
        if self.theorem_radius < 0:
            raise ValueError("theorem_radius must be non-negative.")
        if self.peer_radius < 0:
            raise ValueError("peer_radius must be non-negative.")

    # ------------------------------------------------------------------
    # Geometry & perception (MD-M3: hard step-function boundary)
    # ------------------------------------------------------------------

    def distance_to(self, other_location: np.ndarray) -> float:
        """Euclidean distance to another point. See Theorem MD-T2."""
        return float(np.linalg.norm(self.location - np.asarray(other_location, dtype=float)))

    def can_perceive_theorem(self, theorem_location: np.ndarray) -> bool:
        """True iff theorem is within theorem_radius. MD-M3: hard cutoff."""
        return self.distance_to(theorem_location) <= self.theorem_radius

    def can_perceive_peer(self, peer_location: np.ndarray) -> bool:
        """True iff peer is within peer_radius. MD-M3: hard cutoff."""
        return self.distance_to(peer_location) <= self.peer_radius

    # ------------------------------------------------------------------
    # Communication helpers (MD-G1)
    # ------------------------------------------------------------------

    def packets_to_send(self, neighbor_id: str) -> List[InfoPacket]:
        """
        Return all InfoPackets to send to a given neighbor.

        Includes:
          - self_info (own actions this step)
          - gossip_buffer entries whose generator_id ≠ neighbor_id
            (don't relay a mathematician's actions back to them). MD-G1.
        """
        gossip = [p for p in self.gossip_buffer if p.generator_id != neighbor_id]
        return self.self_info + gossip

    def advance_gossip(self, received_this_step: List[InfoPacket]) -> None:
        """
        Called at end of each step to rotate the gossip buffer.

        MD-G1: received_this_step becomes next step's gossip_buffer.
        self_info is cleared (own actions are fresh each step).
        """
        self.gossip_buffer = list(received_this_step)
        self.self_info = []

    # ------------------------------------------------------------------
    # Belief summary
    # ------------------------------------------------------------------

    def belief_summary(self) -> dict:
        """Return posterior means for all current beliefs."""
        return {
            "theorem_beliefs": {
                tid: {
                    "importance_mean": tb.importance.mu,
                    "importance_std":  tb.importance.std(),
                    "difficulty_mean": tb.difficulty.mu,
                    "difficulty_std":  tb.difficulty.std(),
                }
                for tid, tb in self.theorem_beliefs.items()
            },
            "peer_beliefs": {
                pid: {"ability_mean": pb.mu, "ability_std": pb.std()}
                for pid, pb in self.peer_beliefs.items()
            },
            "known_implications": list(self.known_implications),
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
            f"knows {len(self.theorem_beliefs)} theorems, "
            f"{len(self.peer_beliefs)} peers, "
            f"{len(self.known_implications)} implications)"
        )
