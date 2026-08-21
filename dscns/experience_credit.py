"""Temporal Credit Assignment (v0.5.3 / Phase 5.5).

Core insight: a modification does not always show its full effect immediately.
Round t's modification may cause a cascade:
  - Round t: Modification A → slight improvement
  - Round t+1: Modification B → continued improvement
  - Round t+2: Modification C → FAILURE

Cannot simply attribute failure to C alone.  Must distribute credit across
the temporal window.

    c_t = Credit(Δθ_t, Outcome_{t:t+k})

Architecture:
  CreditSignal      -- per-round credit signal
  ExperienceCreditAssigner -- assigns temporal credit with discount γ
  TemporalCreditTracker -- tracks credit over time
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CreditSignal:
    """Credit signal for one modification at one time step."""
    round_id: int = 0
    immediate_reward: float = 0.0      # r_t (positive for success, negative for failure)
    discounted_return: float = 0.0     # G_t = Σ γ^j r_{t+j}
    cumulative_credit: float = 0.0     # c_t (final credit after attribution)
    credit_window: int = 1             # k used for this computation
    gamma: float = 0.95                # discount factor used
    # component breakdown
    rewards_in_window: List[float] = field(default_factory=list)
    round_ids_in_window: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "immediate_reward": self.immediate_reward,
            "discounted_return": self.discounted_return,
            "cumulative_credit": self.cumulative_credit,
            "credit_window": self.credit_window,
            "gamma": self.gamma,
            "n_rewards": len(self.rewards_in_window),
        }


class ExperienceCreditAssigner:
    """Assign temporal credit to each modification based on future outcomes.

    For each round t with modification Δθ_t, computes:

        G_t(k, γ) = Σ_{j=0}^{k-1} γ^j * r_{t+j}

    where r_{t+j} is the reward at round t+j and γ is the discount factor.

    The credit signal c_t captures how much this modification contributes
    to the system's overall trajectory over the next k rounds.

    This replaces the v0.5.2 model where each modification only had an
    immediate SUCCESS/FAILURE label with no temporal credit.
    """

    def __init__(self, gamma: float = 0.95, default_k: int = 3,
                 credit_scale: float = 1.0):
        """
        Args:
            gamma: discount factor for temporal credit (0 < γ < 1).
            default_k: default temporal credit window size.
            credit_scale: multiplier for credit signals.
        """
        self.gamma = float(np.clip(gamma, 0.01, 0.999))
        self.default_k = int(max(1, default_k))
        self.credit_scale = float(credit_scale)

        # per-round reward storage: round_id -> reward
        self.rewards: Dict[int, float] = {}
        # per-round modification storage: round_id -> modification info
        self.modifications: Dict[int, Dict[str, Any]] = {}
        # credit signals computed
        self.credit_signals: List[CreditSignal] = []
        # history
        self.history: List[Dict[str, Any]] = []

    def record_reward(self, round_id: int, reward: float,
                      modification_info: Optional[Dict[str, Any]] = None) -> None:
        """Record the reward (outcome signal) for a round.

        Args:
            round_id: round number (1-based).
            reward: scalar reward (positive = good, negative = bad).
            modification_info: dict with target, magnitude, direction, etc.
        """
        self.rewards[round_id] = float(reward)
        if modification_info is not None:
            self.modifications[round_id] = modification_info

    def compute_credit(self, round_id: int, k: Optional[int] = None,
                       gamma: Optional[float] = None) -> CreditSignal:
        """Compute temporal credit for the modification at round_id.

        G_t(k, γ) = Σ_{j=0}^{k-1} γ^j * r_{t+j}

        Args:
            round_id: the modification round to compute credit for.
            k: temporal window (uses default_k if None).
            gamma: discount factor (uses self.gamma if None).

        Returns:
            CreditSignal with the computed credit.
        """
        k = k if k is not None else self.default_k
        g = gamma if gamma is not None else self.gamma
        immediate = self.rewards.get(round_id, 0.0)

        # compute discounted return over the window
        discounted_return = 0.0
        rewards_in_window = []
        round_ids_in_window = []
        for j in range(k):
            r_j = self.rewards.get(round_id + j, 0.0)
            discounted_return += (g ** j) * r_j
            rewards_in_window.append(r_j)
            round_ids_in_window.append(round_id + j)

        credit = CreditSignal(
            round_id=round_id,
            immediate_reward=immediate,
            discounted_return=discounted_return,
            cumulative_credit=discounted_return * self.credit_scale,
            credit_window=k,
            gamma=g,
            rewards_in_window=rewards_in_window,
            round_ids_in_window=round_ids_in_window,
        )
        self.credit_signals.append(credit)
        return credit

    def compute_all_credits(self, k: Optional[int] = None,
                            gamma: Optional[float] = None,
                            max_round: Optional[int] = None) -> List[CreditSignal]:
        """Compute credit for all recorded modifications.

        Args:
            k: temporal window size.
            gamma: discount factor.
            max_round: only compute for rounds up to this value.

        Returns:
            List of CreditSignal objects.
        """
        credits = []
        rounds = sorted(self.rewards.keys())
        if max_round is not None:
            rounds = [r for r in rounds if r <= max_round]
        for round_id in rounds:
            credit = self.compute_credit(round_id, k=k, gamma=gamma)
            credits.append(credit)
        return credits

    def compute_multi_window_credits(
        self,
        round_id: int,
        windows: List[int] = None,
        gamma: Optional[float] = None,
    ) -> Dict[int, CreditSignal]:
        """Compute credit at multiple window sizes for comparison.

        Tests: k ∈ {1, 3, 5, 10} per the design spec §38.

        Args:
            round_id: the modification round.
            windows: list of window sizes (default: [1, 3, 5, 10]).
            gamma: discount factor.

        Returns:
            Dict mapping window_size -> CreditSignal.
        """
        if windows is None:
            windows = [1, 3, 5, 10]
        return {k: self.compute_credit(round_id, k=k, gamma=gamma)
                for k in windows}

    def credit_statistics(self) -> Dict[str, Any]:
        """Summary statistics over all computed credits."""
        if not self.credit_signals:
            return {"n_credits": 0}
        credits = [c.cumulative_credit for c in self.credit_signals]
        immediate = [c.immediate_reward for c in self.credit_signals]
        arr = np.array(credits)
        imm = np.array(immediate)
        return {
            "n_credits": len(self.credit_signals),
            "mean_credit": float(arr.mean()),
            "std_credit": float(arr.std()),
            "min_credit": float(arr.min()),
            "max_credit": float(arr.max()),
            "mean_immediate_reward": float(imm.mean()),
            "credit_immediate_correlation": float(np.corrcoef(arr, imm)[0, 1])
                if len(arr) > 1 and arr.std() > 1e-12 and imm.std() > 1e-12 else 0.0,
        }

    def get_credit_for_round(self, round_id: int) -> Optional[CreditSignal]:
        """Get the most recent credit signal for a given round."""
        for c in reversed(self.credit_signals):
            if c.round_id == round_id:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "default_k": self.default_k,
            "n_rewards": len(self.rewards),
            "n_credits": len(self.credit_signals),
            "statistics": self.credit_statistics(),
        }


class TemporalCreditTracker:
    """Track how credit evolves over time and relate to modification outcomes.

    Maintains the credit trajectory for analysis:
      - How does credit change over rounds?
      - Do early modifications receive different credit than late ones?
      - Is there a correlation between credit and future policy changes?
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.credit_history: List[Dict[str, Any]] = []

    def record(self, round_id: int, credit: CreditSignal,
               modification_info: Optional[Dict[str, Any]] = None,
               outcome: str = "neutral") -> None:
        """Record a credit observation."""
        entry = {
            "round_id": round_id,
            "credit": credit.cumulative_credit,
            "immediate_reward": credit.immediate_reward,
            "discounted_return": credit.discounted_return,
            "credit_window": credit.credit_window,
            "gamma": credit.gamma,
            "outcome": outcome,
        }
        if modification_info:
            entry.update(modification_info)
        self.credit_history.append(entry)
        if len(self.credit_history) > self.capacity:
            self.credit_history = self.credit_history[-self.capacity:]

    def credit_trajectory(self) -> List[Dict[str, Any]]:
        """Return the credit trajectory as a list of (round, credit) dicts."""
        return [
            {"round_id": e["round_id"], "credit": e["credit"],
             "outcome": e["outcome"]}
            for e in self.credit_history
        ]

    def credit_by_outcome(self) -> Dict[str, Dict[str, float]]:
        """Mean credit grouped by outcome."""
        by_outcome: Dict[str, List[float]] = {}
        for e in self.credit_history:
            out = e.get("outcome", "neutral")
            by_outcome.setdefault(out, []).append(e["credit"])
        result = {}
        for out, vals in by_outcome.items():
            arr = np.array(vals)
            result[out] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n": len(vals),
            }
        return result

    def credit_trend(self, window: int = 50) -> Dict[str, Any]:
        """Compare mean credit in first half vs second half."""
        n = len(self.credit_history)
        if n < 2:
            return {"trend": 0.0, "early_mean": 0.0, "late_mean": 0.0}
        w = max(1, min(window, n // 2))
        early = np.array([e["credit"] for e in self.credit_history[:w]])
        late = np.array([e["credit"] for e in self.credit_history[-w:]])
        return {
            "trend": float(late.mean() - early.mean()),
            "early_mean": float(early.mean()),
            "late_mean": float(late.mean()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_entries": len(self.credit_history),
            "credit_by_outcome": self.credit_by_outcome(),
            "credit_trend": self.credit_trend(),
        }
