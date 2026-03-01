# Math Importance Model

A toy agent-based model of how mathematicians collectively assign importance to theorems.

Mathematicians and theorems are points in a 2D "mathematical topic space." Mathematicians hold Bayesian beliefs about the difficulty and importance of nearby theorems, and about the ability of nearby peers. Each time step, they attempt proofs, discover implication links, communicate with neighbours, and form an emergent collective view of what matters.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Model Description](#model-description)
3. [Modeling Decisions](#modeling-decisions)  ← **all choices documented here**
4. [Project Structure](#project-structure)
5. [Running the Simulation](#running-the-simulation)

---

## Quickstart

```bash
git clone <repo-url>
cd math-importance-model
pip install -r requirements.txt

# Quick headless check (5 steps, no display needed)
python scripts/demo.py

# Animated simulation (requires a display)
python scripts/run_simulation.py --steps 40

# Headless + save a snapshot image
python scripts/run_simulation.py --steps 50 --no-anim

# Tests
pytest tests/ -v
```

---

## Model Description

### Entities

| Entity | Properties |
|---|---|
| **Theorem** | location ∈ [0,1]², difficulty ∈ ℝ, implication lists |
| **Mathematician** | location ∈ [0,1]², ability ∈ ℝ, theorem_radius, peer_radius, beliefs |

### Mathematical Space

Both entity types live in the unit square **[0,1]×[0,1]**. Proximity represents conceptual similarity.

### Beliefs (all NormalBelief = N(μ, σ²))

Each mathematician maintains **three types** of Bayesian beliefs:

1. **Theorem importance** — for each theorem within `theorem_radius`
2. **Theorem difficulty** — for each theorem within `theorem_radius`
3. **Peer ability** — for each other mathematician within `peer_radius`

All beliefs are **unbounded real**-valued and represented as Normal distributions.
There is **no ground-truth importance** — importance is entirely a collective belief.

### Each Time Step

**Phase 1 — Proof attempts:**
Each mathematician selects the top 2 theorems by importance-belief score, attempts proofs (success probability = sigmoid of ability minus difficulty, penalised by distance), and tries to discover implication links. Beliefs about difficulty update based on success/failure.

**Phase 2 — Communication:**
Each mathematician shares results with her 3 nearest neighbours. Neighbours update beliefs about difficulty, peer ability, and theorem importance. Last step's received info is also gossiped (excluding info back to its source).

**Phase 3 — Spawning:**
New theorems appear at random locations with Gaussian difficulties and form implications with nearby easier theorems. New mathematicians spawn at midpoints between existing mathematicians and their perceived most important theorem.

### Implication Graph

Directed; stored as adjacency lists on each Theorem. Ground truth is fixed once set.
**A less-difficult theorem never implies a more-difficult one** (MD-T6).

---

## Modeling Decisions

> Every explicit choice is tagged `MD-XX` in the source code.
> Change a value → find the tag → understand what breaks/changes.
> Constants are in `model/params.py`; functions are tagged in `model/dynamics.py`.

### Representation

| ID | File | Decision | Alternatives |
|---|---|---|---|
| **MD-T1** | `theorem.py` | Space is **[0,1]×[0,1]** | Higher-dimensional, hyperbolic, discrete |
| **MD-T2** | `theorem.py`, `mathematician.py` | **Euclidean** distance | L1, hyperbolic, cosine |
| **MD-T3** | `theorem.py` | Difficulty is an **unbounded real** | Bounded [0,1], log-normal |
| **MD-T4** | `theorem.py` | **No ground-truth importance** — it is purely a belief | PageRank-like derivation from implication graph |
| **MD-T5** | `theorem.py`, `world.py` | Implication as **directed adjacency lists**; no cycle check | NetworkX graph, weighted edges |
| **MD-T6** | `theorem.py`, `dynamics.py` | **Harder theorems only can imply easier ones** | Allow any direction |
| **MD-M1** | `mathematician.py` | Ability is an **unbounded real** | Bounded [0,1], log-normal |
| **MD-M2** | `mathematician.py` | **Two separate radii**: `theorem_radius` and `peer_radius` | One shared radius |
| **MD-M3** | `mathematician.py` | **Hard step-function** boundary — no belief outside radius | Soft Gaussian kernel decay |
| **MD-M4** | `mathematician.py` | Three belief types treated as **independent** | Joint distribution, copula |
| **MD-M5** | `mathematician.py`, `world.py` | Initial prior **N(0, PRIOR_SIGMA2)** (uninformative) | Informative community prior |
| **MD-W1** | `world.py` | Locations outside [0,1]×[0,1] **rejected** | Toroidal space |
| **MD-W3** | `world.py` | Peer awareness is **asymmetric** | Symmetric mutual awareness |
| **MD-W4** | `world.py` | Implication is **directed; no DAG enforcement** | Enforce DAG |
| **MD-W5** | `world.py` | World tracks **community-level** known implications | Individual only |

### Beliefs

| ID | File | Decision | Alternatives |
|---|---|---|---|
| **MD-B1** | `beliefs.py` | Beliefs are **N(μ, σ²)** over ℝ | Laplace, Student-t, log-Normal |
| **MD-B2** | `beliefs.py` | Default prior **N(0, PRIOR_SIGMA2)** | Jeffreys, informative |
| **MD-B3** | `beliefs.py` | **Normal-Normal conjugate update** (precision-weighted) | Moment matching, variational Bayes |
| **MD-G1** | `mathematician.py`, `dynamics.py` | **Gossip**: relay last-step received info, excluding info back to generator | Track full relay chain; depth cutoff |
| **MD-G2** | `mathematician.py` | Link discovery is of **pre-existing ground-truth implications** | Mathematicians create new implications |

### Dynamics — Phase 1 (Proof Attempts)

| ID | File | Decision | Alternatives |
|---|---|---|---|
| **MD-D1** | `dynamics.py` | Selection score: **importance\_mean / (1 + scale × dist)** | Softmax sampling; include difficulty |
| **MD-D2** | `dynamics.py` | Work on **PROBLEMS\_PER\_STEP = 2** theorems per step | Ability-proportional count |
| **MD-D3** | `dynamics.py` | Success prob: **sigmoid(scale×(ability−diff) − penalty×dist)** using **ground-truth difficulty** | Use belief about difficulty; add noise |
| **MD-D4** | `dynamics.py` | Link discovery: **sigmoid(ability·s − dist\_penalty·(d1+d2) − diff\_penalty·\|gap\| + bias)** | Require working on both theorems |
| **MD-D5** | `dynamics.py` | Difficulty update: **pseudo-obs = ability − dist\_penalty×dist ± failure\_offset**; larger noise on failure | Probit/EP exact update |
| **MD-D6** | `dynamics.py` | On discovering T1→T2: update T2 difficulty toward **T1 difficulty − offset** | No difficulty update from link discovery |

### Dynamics — Phase 2 (Communication)

| ID | File | Decision | Alternatives |
|---|---|---|---|
| **MD-C1** | `dynamics.py` | Communicate with **COMMUNICATION\_PEERS = 3** nearest neighbours | All within radius; random sample |
| **MD-C2** | `dynamics.py` | Peer ability update: **pseudo-obs = difficulty\_belief ± failure\_offset** | Probit likelihood update |
| **MD-C3** | `dynamics.py` | Difficulty update from peer result: use **peer's believed ability** as signal | Use ground truth (information unavailable in practice) |
| **MD-C4** | `dynamics.py` | Importance of X, Y updated when X→Y communicated: **noise ∝ 1/sigmoid(ability\_belief)** | Fixed noise; discount by certainty |

### Dynamics — Phase 3 (Spawning)

| ID | File | Decision | Alternatives |
|---|---|---|---|
| **MD-S1** | `dynamics.py` | Theorem spawn count: **max(1, round(THEOREM\_SPAWN\_RATE × n\_math))** | Fixed count; event-driven |
| **MD-S2** | `dynamics.py` | Mathematician spawn count: **max(0, round(MATHEMATICIAN\_SPAWN\_RATE × n\_math))** | Fixed count |
| **MD-S3** | `dynamics.py` | New theorem location: **Uniform([0,1]²)** | Cluster near active theorems |
| **MD-S4** | `dynamics.py` | New theorem difficulty: **N(DIFFICULTY\_MEAN, DIFFICULTY\_STD²)** | Sample from existing distribution |
| **MD-S5** | `dynamics.py` | Implication probability: **sigmoid(diff\_gap·scale) × exp(−dist·decay)** | Fixed threshold |
| **MD-S6** | `dynamics.py` | New mathematician location: **midpoint(parent, parent's most-believed-important theorem)** | Random; attracted to activity |
| **MD-S7** | `dynamics.py` | New mathematician ability: **N(ABILITY\_MEAN, ABILITY\_STD²)** | Inherit from parent |

### Constants (all in `model/params.py`)

| Param | Default | MD tag | Role |
|---|---|---|---|
| `PRIOR_MU` | 0.0 | MD-C01 | Initial belief mean |
| `PRIOR_SIGMA2` | 4.0 | MD-C02 | Initial belief variance |
| `OBS_NOISE_DIFFICULTY` | 1.0 | MD-C03 | Difficulty obs noise (success) |
| `OBS_NOISE_DIFFICULTY_FAILURE` | 2.0 | MD-C04 | Difficulty obs noise (failure) |
| `OBS_NOISE_ABILITY` | 1.0 | MD-C05 | Peer ability obs noise |
| `OBS_NOISE_IMPORTANCE` | 1.5 | MD-C06 | Importance obs noise |
| `PROBLEMS_PER_STEP` | 2 | MD-C07 | Theorems attempted per step |
| `SELECTION_DISTANCE_SCALE` | 1.0 | MD-C08 | Distance weight in selection |
| `SUCCESS_SCALE` | 1.0 | MD-C09 | Sigmoid scale for success |
| `DISTANCE_PENALTY` | 0.5 | MD-C10 | Distance penalty in success |
| `FAILURE_OFFSET` | 1.0 | MD-C11 | Difficulty shift on failure |
| `LINK_ABILITY_SCALE` | 1.0 | MD-C12 | Ability weight in link discovery |
| `LINK_DISTANCE_PENALTY` | 1.5 | MD-C13 | Distance penalty in link discovery |
| `LINK_DIFF_PENALTY` | 0.5 | MD-C14 | Difficulty gap penalty in link discovery |
| `LINK_BIAS` | -2.0 | MD-C15 | Baseline logit for link discovery |
| `COMMUNICATION_PEERS` | 3 | MD-C16 | Neighbours communicated with |
| `IMPLICATION_BONUS` | 0.5 | MD-C17 | Importance bonus for X when X→Y found |
| `CONSEQUENCE_FACTOR` | 0.7 | MD-C18 | Fraction of X's importance given to Y |
| `THEOREM_SPAWN_RATE` | 0.3 | MD-C19 | New theorems per mathematician per step |
| `MATHEMATICIAN_SPAWN_RATE` | 0.1 | MD-C20 | New mathematicians per existing per step |
| `DIFFICULTY_MEAN` | 0.0 | MD-C21 | Mean of spawned theorem difficulty |
| `DIFFICULTY_STD` | 1.5 | MD-C22 | Std dev of spawned theorem difficulty |
| `ABILITY_MEAN` | 0.0 | MD-C23 | Mean ability of spawned mathematicians |
| `ABILITY_STD` | 1.0 | MD-C24 | Std dev of spawned mathematician ability |
| `THEOREM_RADIUS_DEFAULT` | 0.3 | MD-C25 | Default theorem radius for new mathematicians |
| `PEER_RADIUS_DEFAULT` | 0.4 | MD-C26 | Default peer radius for new mathematicians |
| `IMPLICATION_MAX_DIST` | 0.4 | MD-C27 | Max distance for new implication creation |
| `IMPLICATION_DISTANCE_DECAY` | 4.0 | MD-C28 | Distance decay for implication probability |
| `IMPLICATION_DIFF_SCALE` | 1.0 | MD-C29 | Sigmoid scale for difficulty gap in implication |

---

## Project Structure

```
math-importance-model/
├── README.md                  ← you are here; central modeling-decisions reference
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── __init__.py
│   ├── beliefs.py             ← NormalBelief (MD-B*)
│   ├── theorem.py             ← Theorem (MD-T*)
│   ├── mathematician.py       ← Mathematician, TheoremBeliefs, InfoPacket (MD-M*, MD-G*)
│   ├── world.py               ← MathematicalWorld container (MD-W*)
│   ├── params.py              ← SimParams — ALL constants with MD-Cxx tags
│   ├── dynamics.py            ← Three-phase step function (MD-D*, MD-C*, MD-S*)
│   └── visualization.py       ← Matplotlib plots and FuncAnimation
│
├── scripts/
│   ├── demo.py                ← Headless 5-step quick check
│   └── run_simulation.py      ← Full animated/headless simulation runner
│
└── tests/
    └── test_skeleton.py       ← Structural + dynamics tests
```

---

## Running the Simulation

```bash
# Default: 40 steps, animated
python scripts/run_simulation.py

# More steps, faster animation
python scripts/run_simulation.py --steps 80 --interval 150

# Headless (no display) + save PNG
python scripts/run_simulation.py --steps 50 --no-anim

# Reproduce a specific run
python scripts/run_simulation.py --steps 40 --seed 7
```

Override any constant without touching the code:

```python
from model import SimParams
params = SimParams(
    PROBLEMS_PER_STEP=3,      # each mathematician tries 3 theorems per step
    COMMUNICATION_PEERS=5,    # communicate with 5 nearest neighbours
    THEOREM_SPAWN_RATE=0.5,   # more new theorems each step
)
```
