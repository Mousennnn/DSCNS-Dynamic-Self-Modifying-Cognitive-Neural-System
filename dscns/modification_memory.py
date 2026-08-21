"""Modification memory (Phase 4 / 5.1).

Phase 4: records structural-modification attempts (self_modification.py).

Phase 5.1: adds an episodic self-modification memory for the intrinsic
parameter-modification loop (report sections 10-11).  Each modification
attempt records a full (state, proposal, outcome) transition:

  SUCCESS   — modification improved the probe
  FAILURE   — modification degraded the probe
  RECOVERY  — previous was failure, this one is success

The memory supports similarity-based retrieval (top-k by cosine on the
stored core representation) for error-conditioned modification learning.
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


# ====================================================================== #
# P5.1 Episodic Self-Modification Memory (report sections 10-11)
# ====================================================================== #

@dataclass
class EpisodicModificationRecord:
    """One parameter-modification experience (P5.1).

    Stores the full transition: state + proposal → outcome, so that
    the system can learn state→modification→consequence mappings.
    """
    round_id: int = 0
    core_z: Optional[Any] = None         # (256,) core representation at generation
    state_pooled: Optional[Any] = None   # (768,) pooled hidden state
    meta_info: Optional[Any] = None      # (meta_dim,) meta state
    target_group: int = 0
    target_probs: Optional[Any] = None   # (num_groups,) target distribution
    magnitude: float = 0.0
    magnitude_applied: float = 0.0       # after safety envelope
    confidence: float = 0.0
    delta_norm: float = 0.0              # ||Δθ||
    # outcome
    task_delta: float = 0.0
    probe_delta: float = 0.0
    probe_loss_delta: float = 0.0
    entropy_delta: float = 0.0
    logit_delta: float = 0.0
    # classification
    outcome: str = "neutral"             # success / failure / partial_success / catastrophic / neutral
    category: str = "success"            # success / failure / recovery
    rolled_back: bool = False
    # P5.2 correction
    correction_applied: bool = False
    correction_norm: float = 0.0
    # error representation (computed, not raw)
    error_state: Optional[Any] = None
    error_embedding: Optional[Any] = None  # (error_dim,) after ErrorEncoder
    # reward for plasticity learning
    reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is None or k in ("core_z", "state_pooled", "meta_info",
                                  "target_probs", "error_embedding", "error_state"):
                continue
            d[k] = v.tolist() if hasattr(v, "tolist") else v
        return d


class EpisodicSelfModificationMemory:
    """Episodic memory for the P5.1 intrinsic modification loop.

    Stores full (state, proposal, outcome) transitions, categorized as
    SUCCESS / FAILURE / RECOVERY.  Supports similarity-based retrieval
    using cosine similarity on stored core_z representations.
    """

    def __init__(self, capacity: int = 2000, top_k: int = 8):
        self.capacity = capacity
        self.top_k = top_k
        self.records: List[EpisodicModificationRecord] = []
        self.stats = {
            "success": 0, "failure": 0, "recovery": 0, "total": 0,
        }

    def add(self, record: EpisodicModificationRecord) -> None:
        cat = record.category
        if cat in self.stats:
            self.stats[cat] = self.stats.get(cat, 0) + 1
        self.stats["total"] = self.stats.get("total", 0) + 1
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity:]

    def retrieve_similar(self, query_z: Any, k: Optional[int] = None,
                         category: Optional[str] = None) -> List[EpisodicModificationRecord]:
        """Retrieve top-k records most similar to query_z (cosine sim)."""
        import torch

        k = k or self.top_k
        if not self.records:
            return []
        if query_z is None:
            return self.records[-k:]
        candidates = self.records if category is None else [
            r for r in self.records if r.category == category]
        if not candidates:
            candidates = self.records
        if isinstance(query_z, torch.Tensor):
            query_np = query_z.detach().cpu().float().numpy().flatten()
        else:
            query_np = np.asarray(query_z, dtype=np.float32).flatten()
        sims = []
        for r in candidates:
            if r.core_z is None:
                sims.append((0.0, r))
                continue
            if isinstance(r.core_z, torch.Tensor):
                rz = r.core_z.detach().cpu().float().numpy().flatten()
            else:
                rz = np.asarray(r.core_z, dtype=np.float32).flatten()
            norm = np.linalg.norm(query_np) * np.linalg.norm(rz)
            if norm < 1e-12:
                sims.append((0.0, r))
            else:
                sims.append((float(np.dot(query_np, rz) / norm), r))
        sims.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in sims[:k]]

    def get_category_counts(self) -> Dict[str, int]:
        return dict(self.stats)

    def get_failure_rate(self, window: Optional[int] = None) -> float:
        recs = self.records if window is None else self.records[-window:]
        if not recs:
            return 0.0
        failures = sum(1 for r in recs if r.category == "failure")
        return failures / len(recs)

    def get_repeated_error_rate(self) -> float:
        """Fraction of failures that repeat the same target_group."""
        failures = [r for r in self.records if r.category == "failure"]
        if len(failures) < 2:
            return 0.0
        prev_target = failures[0].target_group
        repeated = 0
        for r in failures[1:]:
            if r.target_group == prev_target:
                repeated += 1
            prev_target = r.target_group
        return repeated / (len(failures) - 1)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_records": len(self.records),
            "categories": dict(self.stats),
            "failure_rate": self.get_failure_rate(),
            "repeated_error_rate": self.get_repeated_error_rate(),
            "records": [r.to_dict() for r in self.records[-100:]],  # last 100
        }
