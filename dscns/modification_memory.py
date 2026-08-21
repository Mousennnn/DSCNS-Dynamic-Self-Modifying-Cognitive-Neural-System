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

    # ================================================================== #
    # v0.5.1 extensions: multi-similarity retrieval & similar failure     #
    # ================================================================== #

    def retrieve_multi_similarity(
        self,
        query_context: Any = None,
        query_proposal: Any = None,
        query_error: Any = None,
        query_target: Optional[int] = None,
        k: Optional[int] = None,
        lambda_context: float = 0.3,
        lambda_proposal: float = 0.3,
        lambda_error: float = 0.2,
        lambda_target: float = 0.2,
    ) -> List[EpisodicModificationRecord]:
        """v0.5.1: retrieve top-k by weighted multi-similarity (task spec §10).

        Similarity = λ_c*S_context + λ_p*S_proposal + λ_e*S_error + λ_t*S_target
        """
        import torch

        k = k or self.top_k
        if not self.records:
            return []

        scored = []
        for r in self.records:
            sim = 0.0
            # context similarity (on core_z)
            if query_context is not None and r.core_z is not None:
                sim += lambda_context * self._cos_sim(query_context, r.core_z)
            # proposal similarity (on delta norms)
            if query_proposal is not None:
                sim += lambda_proposal * self._proposal_sim(query_proposal, r)
            # error similarity (on error_state)
            if query_error is not None and r.error_state is not None:
                sim += lambda_error * self._error_sim(query_error, r.error_state)
            # target similarity
            if query_target is not None:
                sim += lambda_target * (1.0 if query_target == r.target_group else 0.0)
            scored.append((sim, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def find_similar_failures(
        self,
        query_error: Any = None,
        query_context: Any = None,
        similarity_threshold: float = 0.5,
    ) -> List[EpisodicModificationRecord]:
        """v0.5.1: find past failures similar to the current error (task spec §17).

        A 'similar failure' has multi-similarity >= threshold.
        """
        failures = [r for r in self.records if r.category == "failure"]
        if not failures:
            return []

        similar = []
        for r in failures:
            sim = 0.0
            if query_error is not None and r.error_state is not None:
                sim += 0.5 * self._error_sim(query_error, r.error_state)
            if query_context is not None and r.core_z is not None:
                sim += 0.5 * self._cos_sim(query_context, r.core_z)
            if sim >= similarity_threshold:
                similar.append(r)
        return similar

    def get_rfr_target(self) -> float:
        """v0.5.1: Repeat Failure Rate by same target (baseline)."""
        return self.get_repeated_error_rate()

    def get_rfr_similar(self, similarity_threshold: float = 0.5) -> float:
        """v0.5.1: Repeat Failure Rate by similar context/error/proposal (task spec §17).

        Counts consecutive failures where the MULTI-SIMILARITY between
        their error states + contexts exceeds the threshold.
        """
        failures = [r for r in self.records if r.category == "failure"]
        if len(failures) < 2:
            return 0.0
        repeated = 0
        for i in range(1, len(failures)):
            sim = 0.0
            curr, prev = failures[i], failures[i - 1]
            if curr.error_state is not None and prev.error_state is not None:
                sim += 0.5 * self._error_sim(curr.error_state, prev.error_state)
            if curr.core_z is not None and prev.core_z is not None:
                sim += 0.5 * self._cos_sim(curr.core_z, prev.core_z)
            if sim >= similarity_threshold:
                repeated += 1
        return repeated / (len(failures) - 1)

    def get_rfr_exact(self) -> float:
        """v0.5.1: Repeat Failure Rate by exact same conditions."""
        failures = [r for r in self.records if r.category == "failure"]
        if len(failures) < 2:
            return 0.0
        repeated = 0
        for i in range(1, len(failures)):
            curr, prev = failures[i], failures[i - 1]
            if (curr.target_group == prev.target_group and
                    curr.magnitude == prev.magnitude):
                repeated += 1
        return repeated / (len(failures) - 1)

    def get_weight_after_outcome(self, outcome_type: str) -> List[float]:
        """v0.5.1: get weights for records of given category.

        Used to verify: w_failure < w_success (task spec §4).
        """
        return [r.magnitude for r in self.records if r.category == outcome_type]

    def get_weight_stats_by_outcome(self) -> Dict[str, Dict[str, float]]:
        """v0.5.1: weight statistics grouped by outcome category."""
        import torch
        stats = {}
        for cat in ["success", "failure", "recovery"]:
            weights = [r.magnitude for r in self.records if r.category == cat]
            if weights:
                arr = np.array(weights)
                stats[cat] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "count": len(weights),
                }
            else:
                stats[cat] = {"mean": 0.0, "std": 0.0, "min": 0.0,
                              "max": 0.0, "count": 0}
        return stats

    def get_target_transition_matrix(self) -> Dict[str, int]:
        """v0.5.1: count target transitions (e.g., 0→1, 1→2, etc.)."""
        transitions: Dict[str, int] = {}
        prev_target = None
        for r in self.records:
            if prev_target is not None:
                key = f"{prev_target}->{r.target_group}"
                transitions[key] = transitions.get(key, 0) + 1
            prev_target = r.target_group
        return transitions

    @staticmethod
    def _cos_sim(a: Any, b: Any) -> float:
        """Cosine similarity between two tensors/arrays."""
        import torch
        if isinstance(a, torch.Tensor):
            a_np = a.detach().cpu().float().numpy().flatten()
        elif isinstance(a, np.ndarray):
            a_np = a.flatten()
        else:
            return 0.0
        if isinstance(b, torch.Tensor):
            b_np = b.detach().cpu().float().numpy().flatten()
        elif isinstance(b, np.ndarray):
            b_np = b.flatten()
        else:
            return 0.0
        norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if norm < 1e-12:
            return 0.0
        return float(np.dot(a_np, b_np) / norm)

    @staticmethod
    def _proposal_sim(query: Any, record: EpisodicModificationRecord) -> float:
        """Proposal similarity based on delta norms + target."""
        if isinstance(query, dict):
            q_norm = query.get("delta_norm", 0.0)
            q_tgt = query.get("target_group", -1)
        else:
            q_norm = float(query) if query is not None else 0.0
            q_tgt = -1
        norm_sim = 1.0 - min(abs(q_norm - record.delta_norm) / max(q_norm + record.delta_norm, 1e-6), 1.0)
        tgt_sim = 1.0 if q_tgt == record.target_group else 0.0
        return 0.6 * norm_sim + 0.4 * tgt_sim

    @staticmethod
    def _error_sim(a: Any, b: Any) -> float:
        """Error state similarity."""
        import torch
        if hasattr(a, "to_tensor"):
            a = a.to_tensor()
        if hasattr(b, "to_tensor"):
            b = b.to_tensor()
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a_f = a.detach().cpu().float().flatten()
            b_f = b.detach().cpu().float().flatten()
            norm = a_f.norm() * b_f.norm()
            if norm < 1e-12:
                return 0.0
            return float(torch.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)))
        return 0.0
