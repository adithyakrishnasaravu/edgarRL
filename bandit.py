"""
bandit.py — Multi-armed bandit policy for the extraction action space.

Two bandit variants:
  1. FieldBandit     — one bandit per field; learns best action per field
  2. ContextBandit   — one bandit per (field, filer_type); filer_type = SIC bucket

Both use UCB1 (Upper Confidence Bound) by default, which balances
exploration vs exploitation without requiring a tuned epsilon.

The bandit is the Phase 3 "start simple" policy before graduating to PPO.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

N_ACTIONS = 6
FIELD_CONFIG_PATH = Path(__file__).parent / "field_config.json"
with open(FIELD_CONFIG_PATH) as f:
    _CFG = json.load(f)
FIELD_NAMES: list[str] = _CFG["field_order"]


# ---------------------------------------------------------------------------
# UCB1 single bandit
# ---------------------------------------------------------------------------

class UCB1Bandit:
    """
    Upper Confidence Bound (UCB1) bandit for one context.

    Q[a]     = running average reward for action a
    N[a]     = number of times action a was taken
    ucb(a,t) = Q[a] + c * sqrt(ln(t) / N[a])
    """

    def __init__(self, n_actions: int = N_ACTIONS, c: float = 1.0):
        self.n_actions = n_actions
        self.c = c
        self.Q = np.zeros(n_actions, dtype=np.float64)   # mean rewards
        self.N = np.zeros(n_actions, dtype=np.float64)   # visit counts
        self.t = 0                                        # total steps

    def select(self) -> int:
        """Return action index with highest UCB score."""
        self.t += 1
        # Force each action to be tried at least once
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return int(untried[0])

        ucb_scores = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(ucb_scores))

    def update(self, action: int, reward: float) -> None:
        """Incremental mean update."""
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    @property
    def best_action(self) -> int:
        return int(np.argmax(self.Q))

    @property
    def action_stats(self) -> dict:
        return {
            a: {"mean_reward": self.Q[a], "visits": int(self.N[a])}
            for a in range(self.n_actions)
        }

    def __repr__(self) -> str:
        stats = "  ".join(
            f"a{a}:Q={self.Q[a]:+.3f}/N={int(self.N[a])}"
            for a in range(self.n_actions)
        )
        return f"UCB1Bandit(t={self.t}, {stats})"


# ---------------------------------------------------------------------------
# Per-field bandit collection
# ---------------------------------------------------------------------------

SIC_BUCKETS = {
    "tech":         range(7000, 7999),
    "financials":   range(6000, 6999),
    "healthcare":   range(2800, 2999),
    "industrials":  range(3400, 3999),
    "energy":       range(1300, 1499),
    "consumer":     range(2000, 2799),
    "other":        range(0, 9999),   # catch-all
}


def _sic_to_bucket(sic_code: Optional[int]) -> str:
    if sic_code is None:
        return "other"
    for bucket, r in SIC_BUCKETS.items():
        if sic_code in r:
            return bucket
    return "other"


class FieldBandit:
    """
    One UCB1 bandit per field (12 bandits total).
    Learns which extractor works best for each financial field.
    """

    def __init__(self, c: float = 1.0):
        self.bandits: dict[str, UCB1Bandit] = {
            f: UCB1Bandit(N_ACTIONS, c=c) for f in FIELD_NAMES
        }

    def select(self, field_name: str) -> int:
        return self.bandits[field_name].select()

    def update(self, field_name: str, action: int, reward: float) -> None:
        self.bandits[field_name].update(action, reward)

    def best_action(self, field_name: str) -> int:
        return self.bandits[field_name].best_action

    def summary(self) -> dict:
        return {
            field: {
                "best_action": self.bandits[field].best_action,
                "Q": self.bandits[field].Q.tolist(),
                "N": self.bandits[field].N.tolist(),
                "total_steps": self.bandits[field].t,
            }
            for field in FIELD_NAMES
        }


class ContextBandit:
    """
    One UCB1 bandit per (field, sic_bucket) pair.
    More granular: learns e.g. that regex works for tech companies
    but HTML table works better for financials.
    """

    def __init__(self, c: float = 1.0):
        self.c = c
        self._bandits: dict[tuple[str, str], UCB1Bandit] = {}

    def _key(self, field_name: str, sic_code: Optional[int]) -> tuple[str, str]:
        return (field_name, _sic_to_bucket(sic_code))

    def _get(self, key: tuple[str, str]) -> UCB1Bandit:
        if key not in self._bandits:
            self._bandits[key] = UCB1Bandit(N_ACTIONS, c=self.c)
        return self._bandits[key]

    def select(self, field_name: str, sic_code: Optional[int] = None) -> int:
        return self._get(self._key(field_name, sic_code)).select()

    def update(
        self, field_name: str, action: int, reward: float, sic_code: Optional[int] = None
    ) -> None:
        self._get(self._key(field_name, sic_code)).update(action, reward)

    def best_action(self, field_name: str, sic_code: Optional[int] = None) -> int:
        return self._get(self._key(field_name, sic_code)).best_action

    def summary(self) -> dict:
        return {
            f"{field}/{bucket}": {
                "best_action": b.best_action,
                "total_steps": b.t,
                "Q": b.Q.tolist(),
            }
            for (field, bucket), b in self._bandits.items()
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_bandit(bandit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bandit, f)


def load_bandit(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
