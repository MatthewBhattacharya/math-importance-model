"""
Basic tests for the skeleton — verifying structural correctness.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from model import BetaBelief, Theorem, Mathematician, TheoremBeliefs, MathematicalWorld


# -----------------------------------------------------------------------
# BetaBelief
# -----------------------------------------------------------------------

class TestBetaBelief:
    def test_default_is_uniform(self):
        b = BetaBelief()
        assert b.alpha == 1.0 and b.beta_ == 1.0
        assert b.mean() == pytest.approx(0.5)

    def test_update_shifts_mean(self):
        b = BetaBelief()
        b.update(0.0)   # observation = 0
        assert b.mean() < 0.5

        b2 = BetaBelief()
        b2.update(1.0)  # observation = 1
        assert b2.mean() > 0.5

    def test_update_invalid_observation(self):
        b = BetaBelief()
        with pytest.raises(ValueError):
            b.update(1.5)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            BetaBelief(alpha=0, beta_=1)

    def test_copy_is_independent(self):
        b = BetaBelief(alpha=2.0, beta_=3.0)
        c = b.copy()
        c.update(1.0)
        assert b.alpha == 2.0  # original unchanged


# -----------------------------------------------------------------------
# Theorem
# -----------------------------------------------------------------------

class TestTheorem:
    def make(self, **kwargs):
        defaults = dict(
            theorem_id="T", location=np.array([0.5, 0.5]),
            difficulty=0.5, importance=0.5,
        )
        defaults.update(kwargs)
        return Theorem(**defaults)

    def test_basic_creation(self):
        t = self.make()
        assert t.theorem_id == "T"
        assert t.difficulty == 0.5

    def test_out_of_range_difficulty(self):
        with pytest.raises((AssertionError, ValueError)):
            self.make(difficulty=1.5)

    def test_distance(self):
        t = self.make(location=np.array([0.0, 0.0]))
        assert t.distance_to(np.array([1.0, 0.0])) == pytest.approx(1.0)

    def test_implies_lists(self):
        t = self.make()
        t._add_implies("T2")
        t._add_implied_by("T0")
        assert "T2" in t.implies
        assert "T0" in t.implied_by
        # No duplicates
        t._add_implies("T2")
        assert t.implies.count("T2") == 1


# -----------------------------------------------------------------------
# Mathematician
# -----------------------------------------------------------------------

class TestMathematician:
    def make(self, **kwargs):
        defaults = dict(
            mathematician_id="M",
            location=np.array([0.5, 0.5]),
            ability=0.7,
            theorem_radius=0.3,
            peer_radius=0.4,
        )
        defaults.update(kwargs)
        return Mathematician(**defaults)

    def test_basic_creation(self):
        m = self.make()
        assert m.mathematician_id == "M"
        assert m.ability == 0.7

    def test_invalid_ability(self):
        with pytest.raises((AssertionError, ValueError)):
            self.make(ability=1.5)

    def test_perception(self):
        m = self.make(location=np.array([0.5, 0.5]), theorem_radius=0.3)
        # Clearly inside
        assert m.can_perceive_theorem(np.array([0.7, 0.5]))    # dist=0.2
        # Clearly outside
        assert not m.can_perceive_theorem(np.array([0.9, 0.5]))  # dist=0.4


# -----------------------------------------------------------------------
# MathematicalWorld
# -----------------------------------------------------------------------

class TestMathematicalWorld:
    def _theorem(self, tid, loc, diff=0.5, imp=0.5):
        return Theorem(theorem_id=tid, location=np.array(loc), difficulty=diff, importance=imp)

    def _mathematician(self, mid, loc, ability=0.5, t_radius=0.4, p_radius=0.4):
        return Mathematician(
            mathematician_id=mid,
            location=np.array(loc),
            ability=ability,
            theorem_radius=t_radius,
            peer_radius=p_radius,
        )

    def test_add_theorem_and_mathematician(self):
        w = MathematicalWorld()
        w.add_theorem(self._theorem("T1", [0.5, 0.5]))
        w.add_mathematician(self._mathematician("M1", [0.5, 0.5]))
        assert "T1" in w.mathematicians["M1"].theorem_beliefs

    def test_theorem_outside_radius_not_believed(self):
        w = MathematicalWorld()
        w.add_theorem(self._theorem("T1", [0.9, 0.9]))
        w.add_mathematician(self._mathematician("M1", [0.1, 0.1], t_radius=0.1))
        assert "T1" not in w.mathematicians["M1"].theorem_beliefs

    def test_peer_beliefs_added_bidirectionally(self):
        w = MathematicalWorld()
        w.add_mathematician(self._mathematician("A", [0.3, 0.3]))
        w.add_mathematician(self._mathematician("B", [0.4, 0.4]))  # dist ≈ 0.14
        assert "B" in w.mathematicians["A"].peer_beliefs
        assert "A" in w.mathematicians["B"].peer_beliefs

    def test_implication_bidirectional(self):
        w = MathematicalWorld()
        w.add_theorem(self._theorem("T1", [0.2, 0.2]))
        w.add_theorem(self._theorem("T2", [0.8, 0.8]))
        w.add_implication("T1", "T2")
        assert "T2" in w.theorems["T1"].implies
        assert "T1" in w.theorems["T2"].implied_by

    def test_duplicate_theorem_raises(self):
        w = MathematicalWorld()
        w.add_theorem(self._theorem("T1", [0.5, 0.5]))
        with pytest.raises(ValueError):
            w.add_theorem(self._theorem("T1", [0.5, 0.5]))

    def test_location_outside_unit_square_raises(self):
        w = MathematicalWorld()
        with pytest.raises(ValueError):
            w.add_theorem(self._theorem("T", [1.5, 0.5]))

    def test_theorems_near(self):
        w = MathematicalWorld()
        w.add_theorem(self._theorem("T1", [0.1, 0.1]))
        w.add_theorem(self._theorem("T2", [0.9, 0.9]))
        near = w.theorems_near(np.array([0.1, 0.1]), radius=0.2)
        assert len(near) == 1 and near[0].theorem_id == "T1"
