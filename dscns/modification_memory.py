"""Modification memory (Phase 4: learned structural self-adaptation).

Records every structural-modification attempt together with its outcome so
that the learned self-modification policy can revisit past experience
(design-report modification proposal, section 13):

    state_before -> action -> architecture_before/after ->
    short-term gain -> long-term gain -> forgetting change ->
    compute change -> accepted / rejected

The memory doubles as a small replay buffer for the RL stage (REINFORCE)
and as the modification log reported in the experiment results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModificationRecord:
    """One structural-modification experience."""

    round: int
    op: str                     # no_op | expand | contract | split | merge | connect | disconnect
    target: Optional[str] = None
    secondary_target: Optional[str] = None
    magnitude: float = 0.5
    confidence: float = 0.5
    source: str = "rule"        # rule | policy
    state: Any = None           # global self-state vector (numpy, used by RL)
    accepted: Optional[bool] = None
    reward: Optional[float] = None
    probe_before: Optional[float] = None
    probe_after: Optional[float] = None
    delta_perf: Optional[float] = None
    forgetting_change: Optional[float] = None
    param_growth: Optional[float] = None
    compute_cost: Optional[float] = None
    instability: Optional[float] = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "state":
                continue
            d[k] = v.tolist() if hasattr(v, "tolist") else v
        return d


class ModificationMemory:
    """Replay buffer / log of structural-modification experiences."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.records: List[ModificationRecord] = []

    # ------------------------------------------------------------------ #
    def add(self, record: ModificationRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity:]

    # ------------------------------------------------------------------ #
    def structural(self) -> List[ModificationRecord]:
        return [r for r in self.records if r.op != "no_op"]

    def rl_samples(self) -> List[ModificationRecord]:
        """Records carrying a reward signal (used by the RL stage)."""
        return [r for r in self.records if r.reward is not None]

    def success_rate(self, window: Optional[int] = None) -> float:
        recs = self.structural()
        if window:
            recs = recs[-window:]
        decided = [r for r in recs if r.accepted is not None]
        if not decided:
            return 0.0
        return float(np.mean([1.0 if r.accepted else 0.0 for r in decided]))

    def mean_reward(self, window: Optional[int] = None) -> float:
        recs = self.rl_samples()
        if window:
            recs = recs[-window:]
        if not recs:
            return 0.0
        return float(np.mean([float(r.reward) for r in recs]))

    def action_counts(self, source: Optional[str] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            if source is not None and r.source != source:
                continue
            counts[r.op] = counts.get(r.op, 0) + 1
        return counts

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_records": len(self.records),
            "action_counts": self.action_counts(),
            "success_rate": self.success_rate(),
            "mean_reward": self.mean_reward(),
            "records": [r.to_dict() for r in self.records],
        }
