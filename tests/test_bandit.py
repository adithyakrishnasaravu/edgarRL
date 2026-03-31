"""
Tests for bandit.py — UCB1 bandit, FieldBandit, ContextBandit, persistence.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from bandit import (
    UCB1Bandit,
    FieldBandit,
    ContextBandit,
    _sic_to_bucket,
    save_bandit,
    load_bandit,
    N_ACTIONS,
    FIELD_NAMES,
)


# ---------------------------------------------------------------------------
# _sic_to_bucket
# ---------------------------------------------------------------------------

class TestSicToBucket:
    def test_tech(self):
        assert _sic_to_bucket(7372) == "tech"

    def test_financials(self):
        assert _sic_to_bucket(6020) == "financials"

    def test_healthcare(self):
        assert _sic_to_bucket(2834) == "healthcare"

    def test_industrials(self):
        assert _sic_to_bucket(3571) == "industrials"

    def test_energy(self):
        assert _sic_to_bucket(1311) == "energy"

    def test_consumer(self):
        assert _sic_to_bucket(2080) == "consumer"

    def test_none(self):
        assert _sic_to_bucket(None) == "other"

    def test_out_of_range(self):
        # 9999 is in "other" catch-all range(0, 9999) but also in other ranges
        # Actually range(0, 9999) is the catch-all, so anything not matched earlier
        # falls through. Let's test with a value that matches "other" first.
        # The iteration order matters — "tech" is checked first.
        # SIC 500 is in consumer range(2000, 2799)? No, 500 < 2000.
        # 500 is in "other" range(0, 9999)
        assert _sic_to_bucket(500) == "other"


# ---------------------------------------------------------------------------
# UCB1Bandit
# ---------------------------------------------------------------------------

class TestUCB1Bandit:
    def test_construction_defaults(self):
        b = UCB1Bandit()
        assert b.n_actions == N_ACTIONS
        assert b.c == 1.0
        assert b.t == 0
        assert np.all(b.Q == 0.0)
        assert np.all(b.N == 0.0)

    def test_construction_custom(self):
        b = UCB1Bandit(n_actions=3, c=2.0)
        assert b.n_actions == 3
        assert b.c == 2.0

    def test_select_explores_all_first(self):
        b = UCB1Bandit(n_actions=4)
        # Each select returns the first untried action; must update between selects
        # so the action is marked as tried (N[a] > 0)
        actions = []
        for _ in range(4):
            a = b.select()
            actions.append(a)
            b.update(a, 0.0)  # mark as tried
        assert sorted(actions) == [0, 1, 2, 3]

    def test_select_exploits_after_exploration(self):
        b = UCB1Bandit(n_actions=3, c=0.0)  # c=0 → pure exploitation
        # Try each action once
        for _ in range(3):
            a = b.select()
            b.update(a, 0.0)
        # Give action 1 a high reward
        b.update(1, 10.0)
        # With c=0, should always pick action 1 (highest Q)
        assert b.select() == 1

    def test_update_incremental_mean(self):
        b = UCB1Bandit(n_actions=2)
        b.update(0, 1.0)
        assert b.Q[0] == 1.0
        assert b.N[0] == 1

        b.update(0, 3.0)
        assert b.Q[0] == 2.0  # mean of 1.0 and 3.0
        assert b.N[0] == 2

        b.update(0, 5.0)
        assert b.Q[0] == 3.0  # mean of 1, 3, 5
        assert b.N[0] == 3

    def test_best_action(self):
        b = UCB1Bandit(n_actions=3)
        b.Q = np.array([0.1, 0.9, 0.5])
        assert b.best_action == 1

    def test_action_stats(self):
        b = UCB1Bandit(n_actions=2)
        b.update(0, 1.0)
        b.update(1, 2.0)
        stats = b.action_stats
        assert stats[0]["mean_reward"] == 1.0
        assert stats[0]["visits"] == 1
        assert stats[1]["mean_reward"] == 2.0

    def test_repr(self):
        b = UCB1Bandit(n_actions=2)
        b.update(0, 1.0)
        s = repr(b)
        assert "UCB1Bandit" in s
        assert "t=" in s

    def test_ucb_exploration_bonus(self):
        """Action with fewer visits should get higher UCB score."""
        b = UCB1Bandit(n_actions=2, c=1.0)
        # Action 0: visited 100 times, Q=0.5
        b.N[0] = 100
        b.Q[0] = 0.5
        # Action 1: visited 1 time, Q=0.4
        b.N[1] = 1
        b.Q[1] = 0.4
        b.t = 101
        # UCB for action 1 should be higher due to exploration bonus
        selected = b.select()
        assert selected == 1


# ---------------------------------------------------------------------------
# FieldBandit
# ---------------------------------------------------------------------------

class TestFieldBandit:
    def test_construction(self):
        fb = FieldBandit()
        assert len(fb.bandits) == len(FIELD_NAMES)
        for field in FIELD_NAMES:
            assert field in fb.bandits

    def test_select_and_update(self):
        fb = FieldBandit()
        action = fb.select("revenue")
        assert 0 <= action < N_ACTIONS
        fb.update("revenue", action, 1.0)
        assert fb.bandits["revenue"].N[action] == 1

    def test_best_action(self):
        fb = FieldBandit()
        # Give action 2 high reward for revenue
        for _ in range(10):
            fb.update("revenue", 2, 1.0)
        assert fb.best_action("revenue") == 2

    def test_fields_independent(self):
        fb = FieldBandit()
        fb.update("revenue", 0, 1.0)
        fb.update("cogs", 1, 1.0)
        assert fb.bandits["revenue"].N[0] == 1
        assert fb.bandits["revenue"].N[1] == 0
        assert fb.bandits["cogs"].N[1] == 1

    def test_summary(self):
        fb = FieldBandit()
        fb.update("revenue", 0, 1.0)
        summary = fb.summary()
        assert "revenue" in summary
        assert "best_action" in summary["revenue"]
        assert "Q" in summary["revenue"]
        assert "N" in summary["revenue"]
        assert "total_steps" in summary["revenue"]


# ---------------------------------------------------------------------------
# ContextBandit
# ---------------------------------------------------------------------------

class TestContextBandit:
    def test_construction_lazy(self):
        cb = ContextBandit()
        assert len(cb._bandits) == 0

    def test_select_creates_bandit(self):
        cb = ContextBandit()
        action = cb.select("revenue", sic_code=7372)
        assert 0 <= action < N_ACTIONS
        assert len(cb._bandits) == 1

    def test_different_sic_different_bandits(self):
        cb = ContextBandit()
        cb.select("revenue", sic_code=7372)  # tech
        cb.select("revenue", sic_code=6020)  # financials
        assert len(cb._bandits) == 2

    def test_same_field_same_sic_same_bandit(self):
        cb = ContextBandit()
        cb.select("revenue", sic_code=7372)
        cb.select("revenue", sic_code=7500)  # also tech
        assert len(cb._bandits) == 1

    def test_update_routes_correctly(self):
        cb = ContextBandit()
        cb.update("revenue", 0, 1.0, sic_code=7372)
        key = ("revenue", "tech")
        assert key in cb._bandits
        assert cb._bandits[key].N[0] == 1

    def test_best_action(self):
        cb = ContextBandit()
        for _ in range(10):
            cb.update("revenue", 2, 1.0, sic_code=7372)
        assert cb.best_action("revenue", sic_code=7372) == 2

    def test_best_action_no_sic(self):
        cb = ContextBandit()
        for _ in range(10):
            cb.update("revenue", 3, 1.0, sic_code=None)
        assert cb.best_action("revenue", sic_code=None) == 3

    def test_summary(self):
        cb = ContextBandit()
        cb.update("revenue", 0, 1.0, sic_code=7372)
        summary = cb.summary()
        assert "revenue/tech" in summary
        assert "best_action" in summary["revenue/tech"]

    def test_custom_c(self):
        cb = ContextBandit(c=2.0)
        assert cb.c == 2.0
        cb.select("revenue", sic_code=7372)
        key = ("revenue", "tech")
        assert cb._bandits[key].c == 2.0


# ---------------------------------------------------------------------------
# Persistence: save_bandit / load_bandit
# ---------------------------------------------------------------------------

class TestBanditPersistence:
    def test_round_trip_field_bandit(self, tmp_path):
        fb = FieldBandit()
        fb.update("revenue", 0, 1.0)
        fb.update("cogs", 2, 0.5)

        path = tmp_path / "bandits" / "test.pkl"
        save_bandit(fb, path)
        assert path.exists()

        loaded = load_bandit(path)
        assert isinstance(loaded, FieldBandit)
        assert loaded.bandits["revenue"].N[0] == 1
        assert loaded.bandits["cogs"].N[2] == 1

    def test_round_trip_context_bandit(self, tmp_path):
        cb = ContextBandit()
        cb.update("revenue", 0, 1.0, sic_code=7372)

        path = tmp_path / "test_ctx.pkl"
        save_bandit(cb, path)
        loaded = load_bandit(path)
        assert isinstance(loaded, ContextBandit)
        assert ("revenue", "tech") in loaded._bandits

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "bandit.pkl"
        save_bandit(UCB1Bandit(), path)
        assert path.exists()
