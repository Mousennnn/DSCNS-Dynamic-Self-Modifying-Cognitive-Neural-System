"""Policy-to-Modification Trace (v0.6.0 / Phase 6).

Records the complete chain from policy output to actual parameter modification,
enabling causal diagnosis of whether policy changes actually control modifications.

Every round, the trace captures:
    PolicyOutput → Candidate Selection → Proposal → Applied Delta → Outcome

This allows computing:
    - Policy-Action correlation (I(P; A))
    - Target accuracy (does policy's preferred target match actual target?)
    - Magnitude correlation (does policy's magnitude match actual magnitude?)
    - Full chain verification

Components:
  PolicyTraceEntry    -- single-round trace record
  PolicyTraceLog      -- accumulated trace with diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PolicyTraceEntry:
    """Complete trace of one round: policy → modification → outcome."""
    round_id: int = 0
    seed: int = 0
    condition: str = ""

    # state
    state_embedding_norm: float = 0.0
    error_embedding_norm: float = 0.0

    # retrieved experience
    retrieved_experience_ids: List[str] = field(default_factory=list)
    n_retrieved: int = 0

    # policy output
    policy_target_logits: List[float] = field(default_factory=list)
    policy_target_probs: List[float] = field(default_factory=list)
    policy_magnitude: float = 0.0
    policy_confidence: float = 0.0
    policy_direction_mod_norm: float = 0.0
    policy_candidate_scores: List[float] = field(default_factory=list)
    exploration_rate: float = 0.0

    # selected candidate
    selected_candidate_idx: int = 0
    selected_candidate_type: str = ""

    # proposal
    proposal_target: int = -1
    proposal_magnitude: float = 0.0
    proposal_direction_norm: float = 0.0

    # actual applied delta
    actual_target: int = -1
    actual_magnitude: float = 0.0
    actual_delta_norm: float = 0.0
    applied: bool = False

    # outcome
    outcome: str = ""
    reward: float = 0.0
    credit: float = 0.0
    performance_before: float = 0.0
    performance_after: float = 0.0
    delta_performance: float = 0.0

    # parameter state
    param_norm_before: float = 0.0
    param_norm_after: float = 0.0

    # policy learning
    policy_loss: float = 0.0
    policy_kl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "seed": self.seed,
            "condition": self.condition,
            "policy_target_probs": self.policy_target_probs,
            "policy_magnitude": self.policy_magnitude,
            "policy_confidence": self.policy_confidence,
            "exploration_rate": self.exploration_rate,
            "selected_candidate_idx": self.selected_candidate_idx,
            "proposal_target": self.proposal_target,
            "proposal_magnitude": self.proposal_magnitude,
            "actual_target": self.actual_target,
            "actual_magnitude": self.actual_magnitude,
            "actual_delta_norm": self.actual_delta_norm,
            "applied": self.applied,
            "outcome": self.outcome,
            "reward": self.reward,
            "credit": self.credit,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
            "delta_performance": self.delta_performance,
            "param_norm_before": self.param_norm_before,
            "param_norm_after": self.param_norm_after,
            "policy_loss": self.policy_loss,
            "policy_kl": self.policy_kl,
            "n_retrieved": self.n_retrieved,
        }


class PolicyTraceLog:
    """Accumulated trace log with diagnostic capabilities.

    Purpose: Enable causal diagnosis of whether policy output
    actually controls the modification that gets applied.

    Key metrics computed:
        - target_accuracy: how often actual target matches policy's preferred target
        - magnitude_correlation: corr(policy_magnitude, actual_magnitude)
        - applied_ratio: fraction of rounds where modification was applied
        - policy_action_mutual_information: I(P; A)
    """

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.entries: List[PolicyTraceEntry] = []
        self._target_hits = 0
        self._target_misses = 0
        self._magnitude_pairs: List[Tuple[float, float]] = []

    def record(self, entry: PolicyTraceEntry) -> None:
        """Record a trace entry."""
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity:]

        # track target accuracy
        if entry.applied:
            policy_target = int(np.argmax(entry.policy_target_probs)) \
                if entry.policy_target_probs else -1
            if policy_target == entry.actual_target:
                self._target_hits += 1
            else:
                self._target_misses += 1
            self._magnitude_pairs.append(
                (entry.policy_magnitude, entry.actual_magnitude))

    def target_accuracy(self) -> float:
        """Fraction of applied rounds where actual target == policy's argmax target."""
        total = self._target_hits + self._target_misses
        return self._target_hits / max(total, 1)

    def magnitude_correlation(self) -> float:
        """Pearson correlation between policy magnitude and actual magnitude."""
        if len(self._magnitude_pairs) < 5:
            return 0.0
        policy_mags = np.array([p[0] for p in self._magnitude_pairs])
        actual_mags = np.array([p[1] for p in self._magnitude_pairs])
        if policy_mags.std() < 1e-12 or actual_mags.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(policy_mags, actual_mags)[0, 1])

    def applied_ratio(self) -> float:
        """Fraction of rounds where modification was actually applied."""
        if not self.entries:
            return 0.0
        applied = sum(1 for e in self.entries if e.applied)
        return applied / len(self.entries)

    def policy_action_mi(self) -> float:
        """Estimate mutual information between policy target distribution and actual action.

        Uses a simple discrete MI estimate:
            I(P; A) = Σ P(p,a) * log(P(p,a) / (P(p)*P(a)))
        """
        if len(self.entries) < 10:
            return 0.0

        n_targets = 3
        joint = np.zeros((n_targets, n_targets))

        for e in self.entries:
            if not e.applied or not e.policy_target_probs:
                continue
            policy_t = int(np.argmax(e.policy_target_probs))
            actual_t = e.actual_target
            if 0 <= policy_t < n_targets and 0 <= actual_t < n_targets:
                joint[policy_t, actual_t] += 1

        total = joint.sum()
        if total < 5:
            return 0.0

        joint /= total
        marg_p = joint.sum(axis=1)
        marg_a = joint.sum(axis=0)

        mi = 0.0
        for i in range(n_targets):
            for j in range(n_targets):
                if joint[i, j] > 1e-12 and marg_p[i] > 1e-12 and marg_a[j] > 1e-12:
                    mi += joint[i, j] * np.log(joint[i, j] / (marg_p[i] * marg_a[j]))
        return float(mi)

    def outcome_correlation(self) -> Dict[str, float]:
        """Correlation between policy confidence and outcome reward."""
        pairs = [(e.policy_confidence, e.reward) for e in self.entries
                 if e.applied]
        if len(pairs) < 5:
            return {"confidence_reward_corr": 0.0, "n": len(pairs)}
        confs = np.array([p[0] for p in pairs])
        rewards = np.array([p[1] for p in pairs])
        corr = 0.0
        if confs.std() > 1e-12 and rewards.std() > 1e-12:
            corr = float(np.corrcoef(confs, rewards)[0, 1])
        return {"confidence_reward_corr": corr, "n": len(pairs)}

    def diagnostics(self) -> Dict[str, Any]:
        """Full diagnostic summary."""
        target_acc = self.target_accuracy()
        mag_corr = self.magnitude_correlation()
        applied_r = self.applied_ratio()
        mi = self.policy_action_mi()
        oc = self.outcome_correlation()

        # compute reward statistics
        rewards = [e.reward for e in self.entries if e.applied]
        outcomes = [e.outcome for e in self.entries if e.applied]
        outcome_counts = {}
        for o in outcomes:
            outcome_counts[o] = outcome_counts.get(o, 0) + 1

        return {
            "n_entries": len(self.entries),
            "n_applied": sum(1 for e in self.entries if e.applied),
            "target_accuracy": target_acc,
            "magnitude_correlation": mag_corr,
            "applied_ratio": applied_r,
            "policy_action_mi": mi,
            "confidence_reward_corr": oc["confidence_reward_corr"],
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "outcome_distribution": outcome_counts,
        }

    def entries_since(self, round_id: int) -> List[PolicyTraceEntry]:
        """Get entries from round_id onwards."""
        return [e for e in self.entries if e.round_id >= round_id]

    def get_entries_by_outcome(self, outcome: str) -> List[PolicyTraceEntry]:
        """Get entries filtered by outcome."""
        return [e for e in self.entries if e.outcome == outcome]

    def policy_trajectory(self) -> List[Dict[str, Any]]:
        """Return policy target probability trajectory over rounds."""
        return [
            {"round_id": e.round_id, "target_probs": e.policy_target_probs,
             "magnitude": e.policy_magnitude, "confidence": e.policy_confidence}
            for e in self.entries
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnostics": self.diagnostics(),
            "n_entries": len(self.entries),
        }

    def to_jsonl(self) -> List[Dict[str, Any]]:
        """Export all entries as list of dicts for JSONL serialization."""
        return [e.to_dict() for e in self.entries]
