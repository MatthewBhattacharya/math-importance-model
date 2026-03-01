"""
SimParams — all numeric constants in one place.

Every value here is a modeling decision. Change a constant here; find the
corresponding MD-Cxx comment in the source to understand its role.

# ============================================================
# HOW TO READ THIS FILE
# ============================================================
# Each constant has a tag like MD-C01 that cross-references:
#   - This file (canonical value + rationale)
#   - The function/module that uses it (# MD-C01 (constant) comment)
# ============================================================
"""

from dataclasses import dataclass, field


@dataclass
class SimParams:
    """
    All tunable constants for one simulation run.

    Instantiate with defaults:
        params = SimParams()

    Override individual constants:
        params = SimParams(PROBLEMS_PER_STEP=3, SUCCESS_SCALE=2.0)
    """

    # ------------------------------------------------------------------
    # Belief defaults                                             (MD-B*)
    # ------------------------------------------------------------------

    PRIOR_MU: float = 0.0
    # MD-C01: Initial mean for all NormalBelief priors.
    # 0 = "no prior opinion" (neutral/uninformative centre).
    # Alternative: domain-specific estimates.

    PRIOR_SIGMA2: float = 4.0
    # MD-C02: Initial variance for all NormalBelief priors.
    # Large value = high uncertainty / uninformative.
    # Alternative: 1.0 (tighter), or infinity (improper flat prior).

    OBS_NOISE_DIFFICULTY: float = 1.0
    # MD-C03: Noise variance σ² assumed in difficulty observations.
    # A proof attempt gives a noisy signal about true difficulty.
    # Smaller → faster belief convergence; larger → slower, more robust.

    OBS_NOISE_DIFFICULTY_FAILURE: float = 2.0
    # MD-C04: Noise variance for a FAILED proof attempt.
    # Failure is less informative than success (harder to distinguish
    # "just barely failed" from "hopelessly outmatched"), so larger.

    OBS_NOISE_ABILITY: float = 1.0
    # MD-C05: Noise variance in ability observations inferred from a
    # peer's proof success/failure. Higher = slower peer-assessment.

    OBS_NOISE_IMPORTANCE: float = 1.5
    # MD-C06: Noise variance for importance updates triggered by a
    # discovered implication link. Higher = weaker social influence.

    # ------------------------------------------------------------------
    # Problem selection                                           (MD-D1, MD-D2)
    # ------------------------------------------------------------------

    PROBLEMS_PER_STEP: int = 2
    # MD-C07: Number of theorems each mathematician attempts per step.
    # The user specified 2; alternative: proportional to ability.

    SELECTION_DISTANCE_SCALE: float = 1.0
    # MD-C08: Weight of distance in the problem-selection score.
    # Score(T) = importance_belief_mean(T) / (1 + SELECTION_DISTANCE_SCALE * dist)
    # Higher → mathematicians strongly prefer nearby theorems.

    # ------------------------------------------------------------------
    # Success probability                                         (MD-D3)
    # ------------------------------------------------------------------

    SUCCESS_SCALE: float = 1.0
    # MD-C09: Scales the logit argument for proof success:
    # P(success) = sigmoid(SUCCESS_SCALE * (ability − diff_mean) − DISTANCE_PENALTY * dist)
    # Higher → sharper transition between easy and hard theorems.

    DISTANCE_PENALTY: float = 0.5
    # MD-C10: Penalty per unit of distance in the success probability.
    # Models that working far outside your speciality is harder.
    # Set to 0 to make success independent of location.

    # ------------------------------------------------------------------
    # Difficulty belief update after proof attempt                (MD-D5)
    # ------------------------------------------------------------------

    FAILURE_OFFSET: float = 1.0
    # MD-C11: How much harder failure implies the theorem is vs ability.
    # On failure, pseudo-observation = ability − DISTANCE_PENALTY*dist + FAILURE_OFFSET.
    # (Success pseudo-obs = ability − DISTANCE_PENALTY*dist, no offset.)
    # Larger → failures push difficulty beliefs higher.

    # ------------------------------------------------------------------
    # Link discovery                                              (MD-D4)
    # ------------------------------------------------------------------

    LINK_ABILITY_SCALE: float = 1.0
    # MD-C12: Weight of mathematician ability in link-discovery probability.
    # P(discover T1→T2) = sigmoid(
    #     LINK_ABILITY_SCALE * ability
    #   − LINK_DISTANCE_PENALTY * (dist1 + dist2)
    #   − LINK_DIFF_PENALTY * |diff(T1) − diff(T2)|
    #   + LINK_BIAS
    # )

    LINK_DISTANCE_PENALTY: float = 1.5
    # MD-C13: Penalty per unit of total distance (to both theorems) in
    # link-discovery probability. Large → must be very close to discover.

    LINK_DIFF_PENALTY: float = 0.5
    # MD-C14: Penalty for large difficulty difference between theorems.
    # Very different difficulty theorems are harder to connect.

    LINK_BIAS: float = -2.0
    # MD-C15: Baseline logit for link discovery (before ability/distance).
    # Negative → discovery is rare by default; increase to make it easier.

    # ------------------------------------------------------------------
    # Communication                                               (MD-C*)
    # ------------------------------------------------------------------

    COMMUNICATION_PEERS: int = 3
    # MD-C16: Number of nearest neighbours each mathematician communicates
    # with per step. The user specified 3.

    IMPLICATION_BONUS: float = 0.5
    # MD-C17: When X→Y is discovered, the importance pseudo-observation
    # for X is: importance_belief(Y).mu + IMPLICATION_BONUS.
    # Larger → discovering implications boosts X's importance more.

    CONSEQUENCE_FACTOR: float = 0.7
    # MD-C18: When X→Y is discovered, Y's importance pseudo-observation
    # is: importance_belief(X).mu * CONSEQUENCE_FACTOR.
    # < 1 means consequences are deemed slightly less important than their causes.

    # ------------------------------------------------------------------
    # Spawning                                                    (MD-S*)
    # ------------------------------------------------------------------

    THEOREM_SPAWN_RATE: float = 0.3
    # MD-C19: Expected new theorems per mathematician per step.
    # Actual count = max(1, round(THEOREM_SPAWN_RATE * n_mathematicians)).

    MATHEMATICIAN_SPAWN_RATE: float = 0.1
    # MD-C20: Expected new mathematicians per existing mathematician per step.
    # Actual count = max(0, round(...)).

    DIFFICULTY_MEAN: float = 0.0
    # MD-C21: Mean of Gaussian used to sample new theorem difficulties.
    # 0 = centred; theorems are equally likely to be easy or hard.

    DIFFICULTY_STD: float = 1.5
    # MD-C22: Std dev of Gaussian for new theorem difficulties.
    # Larger → wider spread of difficulty.

    ABILITY_MEAN: float = 0.0
    # MD-C23: Mean ability of newly spawned mathematicians.

    ABILITY_STD: float = 1.0
    # MD-C24: Std dev of ability for newly spawned mathematicians.

    THEOREM_RADIUS_DEFAULT: float = 0.3
    # MD-C25: Default theorem_radius for newly spawned mathematicians.

    PEER_RADIUS_DEFAULT: float = 0.4
    # MD-C26: Default peer_radius for newly spawned mathematicians.

    IMPLICATION_MAX_DIST: float = 0.4
    # MD-C27: Max distance between a new theorem and an existing one for
    # a new implication to be considered. Keeps implications "local."

    IMPLICATION_DISTANCE_DECAY: float = 4.0
    # MD-C28: Distance decay rate for new-theorem implication probability.
    # P(link) ∝ exp(−IMPLICATION_DISTANCE_DECAY * distance).
    # Larger → implications only form between very close theorems.

    IMPLICATION_DIFF_SCALE: float = 1.0
    # MD-C29: Sigmoid scale for difficulty difference in implication probability.
    # P(link) ∝ sigmoid(IMPLICATION_DIFF_SCALE * (diff_new − diff_existing)).
    # Larger → implication probability more sensitive to difficulty gap.
