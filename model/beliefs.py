"""
Bayesian belief distributions for the mathematical world model.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-B1: BELIEF REPRESENTATION
#   Each uncertain parameter is represented as a Beta distribution Beta(α, β).
#   Rationale: Beta is the canonical distribution on [0,1], flexible (spans
#   U-shaped, uniform, unimodal), and is the conjugate prior for Bernoulli
#   and Binomial likelihoods, making Bayesian updates analytically clean.
#   Alternatives considered: truncated Normal, Kumaraswamy.
#
# MD-B2: DEFAULT (UNINFORMATIVE) PRIOR
#   Alpha = beta_ = 1.0, giving Beta(1,1) = Uniform[0,1].
#   Rationale: Maximum entropy on [0,1]; no bias before any evidence.
#   Alternative: Jeffrey's prior Beta(0.5, 0.5), which is "less informative"
#   in a formal sense but has mass at the boundaries.
#
# MD-B3: BELIEF UPDATE RULE
#   Given an observation x ∈ [0,1] with confidence weight w:
#       α ← α + w * x
#       β ← β + w * (1 − x)
#   This is exact conjugate Bayesian updating when the likelihood is
#   Bernoulli(x) (treating x as a sufficient statistic). For continuous
#   observations it is a pseudo-count / moment-matching approximation.
#   Alternative: full likelihood specification with a custom update.
# ============================================================
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BetaBelief:
    """
    A Bayesian belief about a single parameter in [0,1], as a Beta(α, β) distribution.

    Attributes:
        alpha:  First shape parameter (α > 0). Encodes pseudo-count of "high" observations.
        beta_:  Second shape parameter (β > 0). Encodes pseudo-count of "low" observations.

    See module docstring for modeling decisions (MD-B1, MD-B2, MD-B3).
    """

    # MD-B2: default uninformative prior Beta(1,1)
    alpha: float = 1.0
    beta_: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta_ <= 0:
            raise ValueError(f"Both alpha and beta_ must be > 0, got α={self.alpha}, β={self.beta_}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def mean(self) -> float:
        """Expected value of the distribution: α / (α + β)."""
        return self.alpha / (self.alpha + self.beta_)

    def variance(self) -> float:
        """Variance: αβ / [(α+β)²(α+β+1)]."""
        a, b = self.alpha, self.beta_
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def mode(self) -> float:
        """
        Mode: (α−1) / (α+β−2), defined only when α > 1 and β > 1.
        Falls back to mean when the mode is undefined (flat or U-shaped distribution).

        # MD-B4 (constant): Fall back to mean — not the boundaries — when mode is
        # undefined, to avoid degenerate point estimates during early simulation steps.
        """
        if self.alpha > 1 and self.beta_ > 1:
            return (self.alpha - 1) / (self.alpha + self.beta_ - 2)
        return self.mean()  # MD-B4 fallback

    def concentration(self) -> float:
        """Total pseudo-count α + β, a proxy for certainty/experience."""
        return self.alpha + self.beta_

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n independent samples from Beta(α, β)."""
        return np.random.beta(self.alpha, self.beta_, size=n)

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update(self, observation: float, weight: float = 1.0) -> None:
        """
        Update belief in-place given a new observation in [0,1].

        Uses the pseudo-count update rule described in MD-B3.

        Args:
            observation: Observed value in [0, 1].
            weight:      Confidence / trust in the observation (default 1.0).
                         # MD-B5 (constant): Default weight = 1.0 (full trust).
                         # Reduce this to model noisy or second-hand evidence.
        """
        if not (0.0 <= observation <= 1.0):
            raise ValueError(f"Observation must be in [0,1], got {observation}")
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        self.alpha += weight * observation
        self.beta_ += weight * (1.0 - observation)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> "BetaBelief":
        """Return a deep copy of this belief."""
        return BetaBelief(alpha=self.alpha, beta_=self.beta_)

    def __repr__(self) -> str:
        return f"Beta(α={self.alpha:.3f}, β={self.beta_:.3f}  mean={self.mean():.3f}, var={self.variance():.4f})"
