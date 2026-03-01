"""
dynamics.py — simulation step logic.

Each call to `step(world, params, rng)` advances the simulation by one time step
through three sequential phases:

  Phase 1 — Proof attempts:
    Each mathematician selects theorems to work on, attempts proofs,
    discovers implication links, and updates her own beliefs.

  Phase 2 — Communication:
    Each mathematician shares her results with nearby peers. Peers update
    their beliefs about difficulty, importance, and peer ability. Gossip
    (relaying last-step info) is also propagated.

  Phase 3 — Spawning:
    New theorems and mathematicians enter the world.

# ============================================================
# MODELING DECISIONS (dynamics-specific)
# ============================================================
#
# --- Phase 1: Problem selection ---
# MD-D1: SELECTION SCORE FUNCTION
#   score(T) = importance_belief_mean(T) / (1 + SELECTION_DISTANCE_SCALE * dist(M, T))
#   Mathematicians prefer theorems they believe important AND nearby.
#   Alternative: softmax sampling instead of argmax; include difficulty in score.
#
# MD-D2: PROBLEMS PER STEP (see params.py MD-C07)
#   Each mathematician works on PROBLEMS_PER_STEP theorems per step (default 2).
#   If fewer are in radius, all available theorems are attempted.
#
# --- Phase 1: Proof success ---
# MD-D3: SUCCESS PROBABILITY
#   P(success | ability a, true_difficulty d, distance dist) =
#       sigmoid(SUCCESS_SCALE * (a − d) − DISTANCE_PENALTY * dist)
#   Success is more likely when ability >> difficulty and mathematician is close.
#   Ground-truth difficulty is used (not the belief), since the theorem's actual
#   hardness determines the outcome, even if the mathematician is surprised by it.
#   Alternative: use sampled difficulty from belief; add noise to ability.
#
# --- Phase 1: Difficulty belief update ---
# MD-D5: PSEUDO-OBSERVATION FOR DIFFICULTY AFTER PROOF ATTEMPT
#   Success: pseudo_obs = ability − DISTANCE_PENALTY * dist
#     (The theorem was solvable at this ability level and distance.)
#   Failure: pseudo_obs = ability − DISTANCE_PENALTY * dist + FAILURE_OFFSET
#     (The theorem proved harder than ability suggests; offset shifts belief up.)
#   Noise: OBS_NOISE_DIFFICULTY (success) or OBS_NOISE_DIFFICULTY_FAILURE (failure).
#   This is a moment-matching approximation; a full probit/logit Bayesian update
#   would require EP or MCMC.
#
# --- Phase 1: Link discovery ---
# MD-D4: LINK DISCOVERY PROBABILITY
#   P(discover T1→T2 | working on T1) =
#       sigmoid(LINK_ABILITY_SCALE * ability
#               − LINK_DISTANCE_PENALTY * (dist(M,T1) + dist(M,T2))
#               − LINK_DIFF_PENALTY * |diff(T1) − diff(T2)|
#               + LINK_BIAS)
#   Conditions: T2 must be in mathematician's theorem_beliefs; T1→T2 must exist
#   in ground truth; the link must not yet be known to the mathematician.
#   Uses ground-truth difficulties (the actual gap determines discoverability).
#   Alternative: use difficulty beliefs; require working on both T1 and T2.
#
# --- Phase 1: Difficulty update from link discovery ---
# MD-D6: IF T1→T2 IS DISCOVERED, UPDATE DIFFICULTY BELIEF FOR T2
#   pseudo_obs = mathematician's own difficulty_belief(T1).mu − LINK_DIFF_PRIOR_OFFSET
#   Rationale: knowing T1 implies T2 (and T1 ≥ T2 in difficulty) provides
#   evidence that T2 is somewhat easier than T1.
#   Alternative: no update; leave T2's difficulty belief unchanged.
#
# --- Phase 2: Peer ability update ---
# MD-C2: PEER ABILITY UPDATE FROM PROOF RESULT
#   When neighbor N sees that M succeeded on T:
#     pseudo_obs for ability(M) = N's belief about difficulty(T).mu
#     (Success means ability is at least as large as difficulty.)
#   When N sees M failed on T:
#     pseudo_obs for ability(M) = N's belief about difficulty(T).mu − FAILURE_OFFSET
#     (Failure means ability is somewhat below difficulty.)
#   Noise: OBS_NOISE_ABILITY.
#   Alternative: use a likelihood function of the probit model for exact Bayesian update.
#
# --- Phase 2: Difficulty update from peer communication ---
# MD-C3: DIFFICULTY UPDATE FROM COMMUNICATED PROOF RESULT
#   When neighbor N hears that M succeeded on T:
#     pseudo_obs for difficulty(T) = N's belief about ability(M).mu − DISTANCE_PENALTY * dist(N,T)
#   When N hears M failed on T:
#     pseudo_obs for difficulty(T) = N's belief about ability(M).mu + FAILURE_OFFSET
#   Noise: OBS_NOISE_DIFFICULTY (success) / OBS_NOISE_DIFFICULTY_FAILURE (failure).
#
# --- Phase 2: Importance update from link discovery ---
# MD-C4: IMPORTANCE UPDATE WHEN X→Y IS COMMUNICATED
#   Pseudo-obs for importance(X) = importance_belief(Y).mu + IMPLICATION_BONUS
#     (X implies Y, so X is at least as important as Y + a bonus.)
#   Pseudo-obs for importance(Y) = importance_belief(X).mu * CONSEQUENCE_FACTOR
#     (Y gains some of X's importance for being a consequence of it.)
#   Noise: OBS_NOISE_IMPORTANCE / (1 + exp(−ability_belief(M).mu))
#     (Higher believed ability of the discoverer → more trustworthy signal.)
#   Alternative: update only X; treat implication as one-directional importance signal.
#
# --- Phase 3: Spawning ---
# MD-S1: THEOREM SPAWN COUNT (see params.py MD-C19)
#   n_new = max(1, round(THEOREM_SPAWN_RATE * n_mathematicians))
#
# MD-S2: MATHEMATICIAN SPAWN COUNT (see params.py MD-C20)
#   n_new = max(0, round(MATHEMATICIAN_SPAWN_RATE * n_mathematicians))
#
# MD-S3: NEW THEOREM LOCATION (see params.py MD-C21, MD-C22)
#   location = Uniform([0,1]²)
#   Alternative: cluster near existing active theorems.
#
# MD-S4: NEW THEOREM DIFFICULTY (see params.py MD-C21, MD-C22)
#   difficulty ~ N(DIFFICULTY_MEAN, DIFFICULTY_STD²)
#   Alternative: sample from existing difficulty distribution.
#
# MD-S5: IMPLICATION CREATION FOR NEW THEOREMS (see params.py MD-C27–MD-C29)
#   For each existing theorem E within IMPLICATION_MAX_DIST of the new theorem T_new:
#     If T_new.difficulty >= E.difficulty (MD-T6: harder can imply easier):
#       P(T_new → E) = sigmoid(IMPLICATION_DIFF_SCALE * (T_new.diff − E.diff))
#                     * exp(−IMPLICATION_DISTANCE_DECAY * dist(T_new, E))
#
# MD-S6: NEW MATHEMATICIAN LOCATION
#   Spawns at midpoint between a randomly chosen existing mathematician M and
#   M's perceived most important theorem (highest importance_belief.mu).
#   If M has no theorem beliefs, spawns at M's location (no-op).
#   Alternative: random location; spawn near highest-traffic theorem.
#
# MD-S7: NEW MATHEMATICIAN ABILITY (see params.py MD-C23, MD-C24)
#   ability ~ N(ABILITY_MEAN, ABILITY_STD²)
#   Alternative: inherit parent's ability with noise.
# ============================================================
"""

import numpy as np
from typing import List, Tuple

from .world import MathematicalWorld
from .theorem import Theorem
from .mathematician import Mathematician, TheoremBeliefs, ProofAttempt, LinkDiscovery, InfoPacket
from .beliefs import NormalBelief
from .params import SimParams


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid σ(x) = 1 / (1 + e^{−x})."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


# ------------------------------------------------------------------
# Phase 1: Proof attempts
# ------------------------------------------------------------------

def _phase_proof_attempts(
    world: MathematicalWorld,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Each mathematician selects theorems, attempts proofs, discovers links,
    and updates her own difficulty beliefs.

    Populates each mathematician's self_info for Phase 2.
    See MD-D1 through MD-D6.
    """
    for m in world.mathematicians.values():
        visible_tids = list(m.theorem_beliefs.keys())
        if not visible_tids:
            continue

        # ---- Problem selection (MD-D1, MD-D2) ----
        def selection_score(tid: str) -> float:
            """MD-D1: score = importance_mean / (1 + SELECTION_DISTANCE_SCALE * dist)"""
            theorem = world.theorems[tid]
            imp = m.theorem_beliefs[tid].importance.mu
            dist = m.distance_to(theorem.location)
            return imp / (1.0 + params.SELECTION_DISTANCE_SCALE * dist)

        sorted_tids = sorted(visible_tids, key=selection_score, reverse=True)
        # MD-D2: work on at most PROBLEMS_PER_STEP theorems
        chosen = sorted_tids[: params.PROBLEMS_PER_STEP]

        for tid in chosen:
            theorem = world.theorems[tid]
            dist = m.distance_to(theorem.location)

            # ---- Success probability (MD-D3) ----
            logit = (
                params.SUCCESS_SCALE * (m.ability - theorem.difficulty)
                - params.DISTANCE_PENALTY * dist
            )
            success = rng.random() < _sigmoid(logit)  # MD-D3

            # Record attempt for communication
            m.self_info.append(InfoPacket(
                content=ProofAttempt(m.mathematician_id, tid, success),
                generator_id=m.mathematician_id,
            ))

            # ---- Difficulty belief update (MD-D5) ----
            baseline = m.ability - params.DISTANCE_PENALTY * dist
            if success:
                pseudo_obs = baseline
                noise = params.OBS_NOISE_DIFFICULTY
            else:
                pseudo_obs = baseline + params.FAILURE_OFFSET  # MD-D5: failure shifts up
                noise = params.OBS_NOISE_DIFFICULTY_FAILURE    # MD-C04: larger noise on failure
            m.theorem_beliefs[tid].difficulty.update(pseudo_obs, obs_noise2=noise)

            # ---- Link discovery (MD-D4) ----
            _attempt_link_discovery(world, m, tid, dist, params, rng)


def _attempt_link_discovery(
    world: MathematicalWorld,
    m: Mathematician,
    worked_on_tid: str,
    dist_to_worked: float,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    While working on `worked_on_tid`, try to discover undiscovered ground-truth
    implications involving it. MD-D4.
    """
    theorem = world.theorems[worked_on_tid]

    # Check all implied theorems (worked_on → other)
    for target_id in theorem.implies:
        _try_discover_link(world, m, worked_on_tid, target_id, dist_to_worked, params, rng)

    # Check all implying theorems (other → worked_on)
    for source_id in theorem.implied_by:
        _try_discover_link(world, m, source_id, worked_on_tid,
                           dist_to_worked, params, rng)


def _try_discover_link(
    world: MathematicalWorld,
    m: Mathematician,
    source_id: str,
    target_id: str,
    dist_to_primary: float,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Try to discover the implication source_id → target_id.

    Conditions for discovery attempt:
      - source_id and target_id must both be in the mathematician's theorem_beliefs.
      - The link must not yet be known to the mathematician.
    """
    if source_id not in m.theorem_beliefs or target_id not in m.theorem_beliefs:
        return
    if (source_id, target_id) in m.known_implications:
        return

    t_source = world.theorems[source_id]
    t_target = world.theorems[target_id]
    dist_other = m.distance_to(t_target.location if target_id != source_id else t_source.location)

    # MD-D4: sigmoid function of ability, total distance, and difficulty gap
    diff_gap = abs(t_source.difficulty - t_target.difficulty)
    logit = (
        params.LINK_ABILITY_SCALE * m.ability
        - params.LINK_DISTANCE_PENALTY * (dist_to_primary + dist_other)
        - params.LINK_DIFF_PENALTY * diff_gap
        + params.LINK_BIAS
    )
    if rng.random() < _sigmoid(logit):
        world.record_discovery(m.mathematician_id, source_id, target_id)
        m.self_info.append(InfoPacket(
            content=LinkDiscovery(m.mathematician_id, source_id, target_id),
            generator_id=m.mathematician_id,
        ))

        # MD-D6: update difficulty belief for the implied theorem
        if target_id in m.theorem_beliefs:
            pseudo_obs = (
                m.theorem_beliefs[source_id].difficulty.mu
                - params.FAILURE_OFFSET  # FAILURE_OFFSET reused as "link diff prior"
            )
            m.theorem_beliefs[target_id].difficulty.update(
                pseudo_obs,
                obs_noise2=params.OBS_NOISE_DIFFICULTY,
            )


# ------------------------------------------------------------------
# Phase 2: Communication
# ------------------------------------------------------------------

def _phase_communication(
    world: MathematicalWorld,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Each mathematician shares results with her COMMUNICATION_PEERS nearest neighbours.
    Neighbours update their beliefs about difficulty, importance, and peer ability.
    Gossip (previous-step received info) is also relayed.

    See MD-C1 through MD-C4, MD-G1.
    """
    # Collect packets received this step for each mathematician
    received: dict[str, list] = {mid: [] for mid in world.mathematicians}

    for m in world.mathematicians.values():
        # MD-C1 (MD-C16): COMMUNICATION_PEERS nearest neighbours
        neighbours = world.nearest_mathematicians(
            m.mathematician_id, params.COMMUNICATION_PEERS
        )

        for n in neighbours:
            packets = m.packets_to_send(n.mathematician_id)  # MD-G1
            for packet in packets:
                _apply_packet_to_peer(world, receiver=n, packet=packet, params=params)
                # Queue for n's gossip buffer
                received[n.mathematician_id].append(packet)

    # MD-G1: rotate gossip buffers at end of communication phase
    for m in world.mathematicians.values():
        m.advance_gossip(received[m.mathematician_id])


def _apply_packet_to_peer(
    world: MathematicalWorld,
    receiver: Mathematician,
    packet: InfoPacket,
    params: SimParams,
) -> None:
    """
    Apply a single InfoPacket to a receiver mathematician's beliefs.

    Dispatches to the appropriate update function based on content type.
    """
    content = packet.content
    if isinstance(content, ProofAttempt):
        _update_from_proof(world, receiver, content, params)
    elif isinstance(content, LinkDiscovery):
        _update_from_link(world, receiver, content, params)


def _update_from_proof(
    world: MathematicalWorld,
    receiver: Mathematician,
    attempt: ProofAttempt,
    params: SimParams,
) -> None:
    """
    Receiver updates beliefs about theorem difficulty and peer ability
    given knowledge that mathematician `attempt.mathematician_id` succeeded
    or failed on theorem `attempt.theorem_id`. MD-C2, MD-C3.
    """
    tid = attempt.theorem_id
    mid = attempt.mathematician_id
    success = attempt.success

    # ---- Difficulty update (MD-C3) ----
    # Only possible if receiver knows this theorem
    if tid in receiver.theorem_beliefs:
        # Use receiver's belief about the prover's ability as the signal
        prover_ability_mu = (
            receiver.peer_beliefs[mid].mu
            if mid in receiver.peer_beliefs
            else params.PRIOR_MU
        )
        theorem = world.theorems.get(tid)
        dist_r = receiver.distance_to(theorem.location) if theorem else 0.0

        if success:
            # MD-C3: success implies difficulty ≈ prover_ability − distance_effect
            pseudo_obs = prover_ability_mu - params.DISTANCE_PENALTY * dist_r
            noise = params.OBS_NOISE_DIFFICULTY
        else:
            pseudo_obs = prover_ability_mu + params.FAILURE_OFFSET
            noise = params.OBS_NOISE_DIFFICULTY_FAILURE

        receiver.theorem_beliefs[tid].difficulty.update(pseudo_obs, obs_noise2=noise)

    # ---- Peer ability update (MD-C2) ----
    if mid in receiver.peer_beliefs:
        # Use receiver's belief about the theorem's difficulty as the signal
        diff_mu = (
            receiver.theorem_beliefs[tid].difficulty.mu
            if tid in receiver.theorem_beliefs
            else params.PRIOR_MU
        )
        if success:
            pseudo_obs = diff_mu
        else:
            pseudo_obs = diff_mu - params.FAILURE_OFFSET

        receiver.peer_beliefs[mid].update(pseudo_obs, obs_noise2=params.OBS_NOISE_ABILITY)


def _update_from_link(
    world: MathematicalWorld,
    receiver: Mathematician,
    link: LinkDiscovery,
    params: SimParams,
) -> None:
    """
    Receiver updates beliefs about the importance of both theorems in
    the newly communicated implication link source → target. MD-C4.

    Also records the link as known to the receiver.
    """
    sid = link.source_id
    tid = link.target_id
    mid = link.mathematician_id

    # Record discovery for receiver too
    if (sid, tid) not in receiver.known_implications:
        receiver.known_implications.add((sid, tid))
        world.known_implications.add((sid, tid))

    # ---- Importance update (MD-C4) ----
    # Noise is modulated by believed ability of the discoverer
    # Higher believed ability → more trustworthy → lower noise
    ability_mu = (
        receiver.peer_beliefs[mid].mu
        if mid in receiver.peer_beliefs
        else params.PRIOR_MU
    )
    # Transform ability into a noise scaling: more capable → less noise
    # MD-C4: noise = OBS_NOISE_IMPORTANCE / sigmoid(ability_mu)
    # sigmoid maps ability to (0,1), avoiding division by zero
    noise = params.OBS_NOISE_IMPORTANCE / max(_sigmoid(ability_mu), 0.05)

    if sid in receiver.theorem_beliefs and tid in receiver.theorem_beliefs:
        imp_source = receiver.theorem_beliefs[sid].importance.mu
        imp_target = receiver.theorem_beliefs[tid].importance.mu

        # X→Y discovered: X is at least as important as Y (+ bonus)
        pseudo_source = imp_target + params.IMPLICATION_BONUS  # MD-C17
        receiver.theorem_beliefs[sid].importance.update(pseudo_source, obs_noise2=noise)

        # Y gains some of X's importance (weighted by CONSEQUENCE_FACTOR)
        pseudo_target = imp_source * params.CONSEQUENCE_FACTOR  # MD-C18
        receiver.theorem_beliefs[tid].importance.update(pseudo_target, obs_noise2=noise)

    elif sid in receiver.theorem_beliefs:
        # Receiver only knows source — small boost from knowing it implies something
        imp_source = receiver.theorem_beliefs[sid].importance.mu
        pseudo_source = imp_source + params.IMPLICATION_BONUS
        receiver.theorem_beliefs[sid].importance.update(pseudo_source, obs_noise2=noise)

    elif tid in receiver.theorem_beliefs:
        # Receiver only knows target — small boost from knowing something implies it
        imp_target = receiver.theorem_beliefs[tid].importance.mu
        pseudo_target = imp_target + params.IMPLICATION_BONUS * params.CONSEQUENCE_FACTOR
        receiver.theorem_beliefs[tid].importance.update(pseudo_target, obs_noise2=noise)


# ------------------------------------------------------------------
# Phase 3: Spawning
# ------------------------------------------------------------------

def _phase_spawn(
    world: MathematicalWorld,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Spawn new theorems and mathematicians. MD-S1 through MD-S7.
    """
    _spawn_theorems(world, params, rng)
    _spawn_mathematicians(world, params, rng)


def _spawn_theorems(
    world: MathematicalWorld,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Spawn new theorems at random locations with Gaussian difficulties.
    Create ground-truth implications to nearby easier theorems. MD-S1, MD-S3–MD-S5.
    """
    n_math = len(world.mathematicians)
    # MD-S1 (MD-C19): spawn count scales with number of mathematicians
    n_new = max(1, round(params.THEOREM_SPAWN_RATE * max(n_math, 1)))

    for _ in range(n_new):
        tid = world.next_theorem_id()

        # MD-S3: location Uniform([0,1]²)
        loc = rng.uniform(0.0, 1.0, size=2)

        # MD-S4: difficulty ~ N(DIFFICULTY_MEAN, DIFFICULTY_STD²)
        diff = float(rng.normal(params.DIFFICULTY_MEAN, params.DIFFICULTY_STD))

        theorem = Theorem(theorem_id=tid, location=loc, difficulty=diff)
        world.add_theorem(theorem)

        # MD-S5: create implications to nearby existing theorems
        for existing in list(world.theorems.values()):
            if existing.theorem_id == tid:
                continue
            dist = theorem.distance_to(existing.location)
            if dist > params.IMPLICATION_MAX_DIST:
                continue

            # MD-T6: only harder theorem can imply easier one
            diff_gap = theorem.difficulty - existing.difficulty
            if diff_gap < 0:
                continue  # new theorem is easier; cannot imply existing

            # MD-S5: P(implication) = sigmoid(DIFF_SCALE * gap) * exp(−DIST_DECAY * dist)
            p = (
                _sigmoid(params.IMPLICATION_DIFF_SCALE * diff_gap)
                * np.exp(-params.IMPLICATION_DISTANCE_DECAY * dist)
            )
            if rng.random() < p:
                world.add_implication(tid, existing.theorem_id)


def _spawn_mathematicians(
    world: MathematicalWorld,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Spawn new mathematicians at midpoints between existing mathematicians
    and their most-believed-important theorem. MD-S2, MD-S6, MD-S7.
    """
    n_math = len(world.mathematicians)
    if n_math == 0:
        return

    # MD-S2 (MD-C20): spawn count
    n_new = max(0, round(params.MATHEMATICIAN_SPAWN_RATE * n_math))

    existing_list = list(world.mathematicians.values())

    for _ in range(n_new):
        # MD-S6: pick a random existing mathematician
        parent = rng.choice(existing_list)  # type: ignore[arg-type]

        if not parent.theorem_beliefs:
            loc = parent.location.copy()
        else:
            # Find the theorem this mathematician believes is most important
            best_tid = max(
                parent.theorem_beliefs,
                key=lambda t: parent.theorem_beliefs[t].importance.mu,
            )
            best_loc = world.theorems[best_tid].location
            # MD-S6: midpoint between parent and their most important theorem
            loc = (parent.location + best_loc) / 2.0
            loc = np.clip(loc, 0.0, 1.0)  # ensure in [0,1]²

        # MD-S7: ability ~ N(ABILITY_MEAN, ABILITY_STD²)
        ability = float(rng.normal(params.ABILITY_MEAN, params.ABILITY_STD))

        mid = world.next_mathematician_id()
        mathematician = Mathematician(
            mathematician_id=mid,
            location=loc,
            ability=ability,
            theorem_radius=params.THEOREM_RADIUS_DEFAULT,   # MD-C25
            peer_radius=params.PEER_RADIUS_DEFAULT,         # MD-C26
        )
        world.add_mathematician(mathematician)


# ------------------------------------------------------------------
# Top-level step
# ------------------------------------------------------------------

def step(
    world: MathematicalWorld,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """
    Advance the simulation by one time step.

    Executes Phase 1 → Phase 2 → Phase 3 in order.
    """
    _phase_proof_attempts(world, params, rng)
    _phase_communication(world, params, rng)
    _phase_spawn(world, params, rng)
