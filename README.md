# Math Importance Model

A toy agent-based model of how mathematicians collectively assign importance to theorems.

Mathematicians and theorems are points in a 2D "mathematical topic space." Mathematicians hold Bayesian beliefs about the difficulty and importance of nearby theorems, and about the ability of nearby peers. The simulation will explore how these beliefs propagate and converge (or fail to).

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Model Description](#model-description)
3. [Modeling Decisions](#modeling-decisions)  ← **all choices documented here**
4. [Project Structure](#project-structure)
5. [Roadmap / Next Steps](#roadmap--next-steps)

---

## Quickstart

```bash
git clone <repo-url>
cd math-importance-model
pip install -r requirements.txt

# Run the demo
python scripts/demo.py

# Run tests
pytest tests/
```

---

## Model Description

### Entities

| Entity | Properties |
|---|---|
| **Theorem** | location ∈ [0,1]², difficulty ∈ [0,1], importance ∈ [0,1], implication lists |
| **Mathematician** | location ∈ [0,1]², ability ∈ [0,1], theorem_radius, peer_radius, beliefs |

### Mathematical Space

Both entity types live in the unit square **[0,1]×[0,1]**. Proximity represents conceptual similarity — two theorems in the same corner of the space are in related areas of mathematics.

### Beliefs

Each mathematician maintains **three types of Bayesian beliefs**, all represented as **Beta distributions** over [0,1]:

1. **Theorem importance** — for each theorem within `theorem_radius`
2. **Theorem difficulty** — for each theorem within `theorem_radius`
3. **Peer ability** — for each other mathematician within `peer_radius`

Mathematicians have **no beliefs** about entities outside their respective radii.

### Implication Graph

Theorems are connected by a directed implication relation (intended to form a DAG).
`add_implication("T1", "T2")` means T1 → T2 (T1 implies T2).

---

## Modeling Decisions

> Every explicit choice in this model is tagged with an `MD-XX` code in the source code.
> This section is the single reference listing all of them.
> When you change a decision, update both this table and the corresponding `# MD-XX` comment.

### Representation

| ID | Location | Decision | Alternatives |
|---|---|---|---|
| **MD-T1** | `theorem.py` | Mathematical space is **[0,1]×[0,1]** | Higher-dimensional, hyperbolic, discrete topic categories |
| **MD-T2** | `theorem.py`, `mathematician.py` | Distance metric is **Euclidean** | L1 (Manhattan), hyperbolic, cosine similarity |
| **MD-T3** | `theorem.py` | Difficulty ∈ **[0,1]** (bounded) | Unbounded positive real (log-normal), ordinal |
| **MD-T4** | `theorem.py` | Importance ∈ **[0,1]**, **latent** ground truth | Emerges from implication structure (PageRank-like) |
| **MD-T5** | `theorem.py`, `world.py` | Implication stored as **adjacency lists** on each Theorem; no cycle detection | Separate graph object (NetworkX), weighted edges |
| **MD-M1** | `mathematician.py` | Ability ∈ **[0,1]** (bounded) | Unbounded real, log-normal |
| **MD-M2** | `mathematician.py` | **Two separate radii**: `theorem_radius` and `peer_radius` | Single shared radius |
| **MD-M3** | `mathematician.py` | **Hard step-function boundary** — no belief outside radius | Soft Gaussian kernel decay with distance |
| **MD-M4** | `mathematician.py` | Three belief types treated as **independent** | Joint distribution (bivariate Beta, copula) |
| **MD-W1** | `world.py` | Locations outside [0,1]×[0,1] are **rejected** | Toroidal space (wrap-around), larger bounding box |
| **MD-W3** | `world.py` | Peer awareness is **asymmetric** (P aware of Q if Q ∈ P's radius, regardless of Q's radius) | Symmetric mutual awareness |
| **MD-W4** | `world.py` | Implication is **directed**; no DAG enforcement | Enforce DAG by topological sort |

### Beliefs

| ID | Location | Decision | Alternatives |
|---|---|---|---|
| **MD-B1** | `beliefs.py` | Beliefs are **Beta(α, β) distributions** over [0,1] | Truncated Normal, Kumaraswamy, Dirichlet |
| **MD-B2** | `beliefs.py` | Default (uninformative) prior is **Beta(1,1) = Uniform[0,1]** | Jeffreys prior Beta(0.5, 0.5) |
| **MD-B3** | `beliefs.py` | Update rule: **pseudo-count** α += w·x, β += w·(1−x) | Full likelihood specification, kernel density update |
| **MD-B4** | `beliefs.py` | When mode undefined (α≤1 or β≤1): **fall back to mean** | Fall back to boundary (0 or 1) |
| **MD-M5** | `mathematician.py`, `world.py` | Initial prior on encounter is **Beta(1,1)** (uninformative) | Informative prior from community statistics |

### Constants

> These are numbers baked into the model that could reasonably be different.
> Each is marked `# MD-Cxx (constant)` in the source code.

| ID | Location | Value | Meaning | Alternatives |
|---|---|---|---|---|
| **MD-B5** | `beliefs.py` | `weight=1.0` (default) | Default trust in a single observation | Lower values model noisy evidence |
| **MD-W2** | `world.py` | Beliefs initialised **immediately** on entity arrival | When to initialise beliefs | Lazy (at first time-step interaction) |

---

## Project Structure

```
math-importance-model/
├── README.md             ← you are here; central modeling-decisions reference
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── __init__.py
│   ├── beliefs.py        ← BetaBelief class (MD-B*)
│   ├── theorem.py        ← Theorem dataclass (MD-T*)
│   ├── mathematician.py  ← Mathematician class (MD-M*)
│   └── world.py          ← MathematicalWorld container (MD-W*)
│
├── scripts/
│   └── demo.py           ← minimal working demonstration
│
└── tests/
    └── test_skeleton.py  ← structural / correctness tests
```

---

## Roadmap / Next Steps

The skeleton defines the entities and belief representations.
The simulation dynamics (to be specified) will add:

- A **time-step loop**: what happens each step (proof attempts, communication, etc.)
- **Belief updates**: how mathematicians update beliefs after observations
- **Communication rules**: how mathematicians share beliefs with peers
- **Metrics**: convergence of community importance estimates to ground truth
