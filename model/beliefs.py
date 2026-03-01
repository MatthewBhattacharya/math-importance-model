"""
NormalBelief — Bayesian belief about an unbounded real-valued parameter.

# ============================================================
# MODELING DECISIONS
# ============================================================
# MD-B1: BELIEF REPRESENTATION
#   Each uncertain parameter (theorem difficulty, theorem importance,
#   peer ability) is represented as a Normal distribution N(μ, σ²).
#   Rationale: natural distribution on ℝ; conjugate to a Gaussian
#   likelihood; μ and σ² are directly interpretable as "best estimate"
#   and "uncertainty". Supports negative values (e.g. negative difficulty
#   meaning "trivially easy").
#   Alternatives: Laplace, Student-t (heavier tails), log-Normal (for
#   strictly positive parameters).
#
# MD-B2: DEFAULT (UNINFORMATIVE) PRIOR
#   μ₀ = PRIOR_MU (default 0.0), σ₀² = PRIOR_SIGMA2 (default 4.0).
#   A large variance represents near-ignorance before any evidence.
#   Alternative: Jeffreys prior (improper flat), or informative priors
#   derived from community statistics.
#
# MD-B3: UPDATE RULE — Normal-Normal conjugate
#   Given an observation x with known noise variance σ_obs²:
#       precision_prior = 1 / σ_prior²
#       precision_obs   = 1 / σ_obs²
#       precision_post  = precision_prior + precision_obs
#       μ_post          = (precision_prior·μ_prior + precision_obs·x) / precision_post
#       σ²_post         = 1 / precision_post
#   This is the exact conjugate Bayesian update for a Gaussian likelihood
#   with known noise. Observations are precision-weighted: a precise
#   (low-noise) observation shifts the mean more than a noisy one.
#   Alternative: moment-matching, variational Bayes, Kalman filter.
# ============================================================
"""

import numpy as np
from dataclasses import dataclass

# Defaults — see MD-B2 and params.py MD-C01, MD-C02
_DEFAULT_MU: float = 0.0
_DEFAULT_SIGMA2: float = 4.0


@dataclass
class NormalBelief:
    """
    A Bayesian belief about a single unbounded real parameter, stored as N(μ, σ²).

    Attributes:
        mu:     Posterior mean — current best estimate of the parameter.
        sigma2: Posterior variance — uncertainty (must be > 0).

    See module docstring for MD-B1, MD-B2, MD-B3.
    """

    mu: float = _DEFAULT_MU       # MD-B2: initial mean (see also params.py MD-C01)
    sigma2: float = _DEFAULT_SIGMA2  # MD-B2: initial variance (see also params.py MD-C02)

    def __post_init__(self) -> None:
        if self.sigma2 <= 0:
            raise ValueError(f"sigma2 must be > 0, got {self.sigma2}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def mean(self) -> float:
        """Best estimate of the parameter (posterior mean)."""
        return self.mu

    def variance(self) -> float:
        """Posterior variance — proxy for remaining uncertainty."""
        return self.sigma2

    def std(self) -> float:
        """Posterior standard deviation."""
        return float(np.sqrt(self.sigma2))

    def precision(self) -> float:
        """Posterior precision = 1/σ². Higher → more certain."""
        return 1.0 / self.sigma2

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n independent samples from N(μ, σ²)."""
        return np.random.normal(self.mu, self.std(), size=n)

    # ------------------------------------------------------------------
    # Bayesian update (MD-B3)
    # ------------------------------------------------------------------

    def update(self, observation: float, obs_noise2: float) -> None:
        """
        In-place Normal-Normal conjugate update.

        Args:
            observation: Observed value (unbounded real).
            obs_noise2:  Known observation noise variance σ_obs² > 0.
                         Smaller → observation is trusted more.
                         See params.py MD-C03 through MD-C06 for default values.

        Raises:
            ValueError: if obs_noise2 ≤ 0.
        """
        if obs_noise2 <= 0:
            raise ValueError(f"obs_noise2 must be > 0, got {obs_noise2}")
        prec_prior = 1.0 / self.sigma2
        prec_obs = 1.0 / obs_noise2
        prec_post = prec_prior + prec_obs
        self.mu = (prec_prior * self.mu + prec_obs * observation) / prec_post
        self.sigma2 = 1.0 / prec_post

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> "NormalBelief":
        """Return an independent copy."""
        return NormalBelief(mu=self.mu, sigma2=self.sigma2)

    def __repr__(self) -> str:
        return f"Normal(μ={self.mu:.3f}, σ={self.std():.3f})"
