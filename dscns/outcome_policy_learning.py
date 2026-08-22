"""Outcome-Directed Policy Learning (v0.6.0 / Phase 6).

Core insight from v0.5.3 failure analysis:
    Memory → Policy coupling was established (D_policy > 0),
    but Policy change did NOT produce better outcomes.

Root cause hypothesis: The policy learning signal was too weak.
v0.5.3 used only:
    L_contrastive + L_avoid + L_reuse + L_stability
These losses do NOT directly tie policy to outcome quality.

v0.6.0 adds outcome-directed reward:
    R_t = w_p * R_performance + w_e * R_error + w_s * R_stability + w_c * R_consistency

Where:
    R_performance = Performance(θ_t) - Performance(θ_{t-1})   [delta, not absolute]
    R_error = Error(θ_{t-1}) - Error(θ_t)                     [error reduction]
    R_stability = -||θ_t - θ_{stable}||                       [penalty for drift]
    R_consistency = consistency of improvement over window

Components:
  ModificationReward        -- unified reward signal
  OutcomeDirectedPolicyLearner -- policy learner using outcome reward
  PolicyCreditAssigner      -- assigns credit to policy actions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ModificationReward:
    """Unified reward for one modification event."""
    round_id: int = 0
    # components
    r_performance: float = 0.0    # delta performance
    r_error: float = 0.0          # error reduction
    r_stability: float = 0.0      # drift penalty
    r_consistency: float = 0.0    # consistency of improvement
    # weighted total
    total: float = 0.0
    # weights used
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "r_performance": self.r_performance,
            "r_error": self.r_error,
            "r_stability": self.r_stability,
            "r_consistency": self.r_consistency,
            "total": self.total,
            "weights": self.weights,
        }


class ModificationRewardModel:
    """Compute unified modification reward.

    R_t = w_p * R_performance + w_e * R_error + w_s * R_stability + w_c * R_consistency

    Key design principle: Uses DELTA, not absolute values.
        R_performance = Performance(θ_t) - Performance(θ_{t-1})
        NOT: R_performance = Performance(θ_t)

    Also maintains baseline-relative reward:
        R_baseline = Performance(θ_t) - Performance(θ_baseline)

    This separates:
        1. Model was already good (absolute performance high)
        2. Modification made it better (delta positive)
    """

    def __init__(
        self,
        w_performance: float = 0.4,
        w_error: float = 0.3,
        w_stability: float = 0.2,
        w_consistency: float = 0.1,
        drift_penalty_scale: float = 0.1,
        consistency_window: int = 10,
    ):
        self.w_performance = w_performance
        self.w_error = w_error
        self.w_stability = w_stability
        self.w_consistency = w_consistency
        self.drift_penalty_scale = drift_penalty_scale
        self.consistency_window = consistency_window

        # state
        self.prev_performance: float = 0.0
        self.baseline_performance: float = 0.0
        self.param_norm_baseline: float = 0.0
        self.recent_performance: List[float] = []
        self.rewards: List[ModificationReward] = []

    def set_baseline(self, performance: float, param_norm: float = 0.0) -> None:
        """Set the initial baseline before any modifications."""
        self.baseline_performance = performance
        self.prev_performance = performance
        self.param_norm_baseline = param_norm

    def compute_reward(
        self,
        round_id: int,
        performance_before: float,
        performance_after: float,
        param_norm_before: float,
        param_norm_after: float,
        error_before: float = 0.0,
        error_after: float = 0.0,
    ) -> ModificationReward:
        """Compute the unified reward for a modification.

        Args:
            round_id: current round.
            performance_before: probe performance before modification.
            performance_after: probe performance after modification.
            param_norm_before: parameter norm before modification.
            param_norm_after: parameter norm after modification.
            error_before: error metric before.
            error_after: error metric after.

        Returns:
            ModificationReward with all components.
        """
        # R_performance: delta (NOT absolute)
        r_perf = performance_after - performance_before

        # R_error: error reduction (positive = error went down)
        r_err = error_before - error_after

        # R_stability: penalize large drift from baseline
        drift = abs(param_norm_after - self.param_norm_baseline)
        r_stab = -self.drift_penalty_scale * drift

        # R_consistency: fraction of recent rounds with positive delta
        self.recent_performance.append(r_perf)
        if len(self.recent_performance) > self.consistency_window:
            self.recent_performance = self.recent_performance[-self.consistency_window:]
        if len(self.recent_performance) >= 3:
            r_cons = sum(1 for p in self.recent_performance if p > 0) / len(self.recent_performance)
            # scale to [-1, 1]
            r_cons = 2.0 * r_cons - 1.0
        else:
            r_cons = 0.0

        # weighted total
        total = (self.w_performance * r_perf +
                 self.w_error * r_err +
                 self.w_stability * r_stab +
                 self.w_consistency * r_cons)

        reward = ModificationReward(
            round_id=round_id,
            r_performance=r_perf,
            r_error=r_err,
            r_stability=r_stab,
            r_consistency=r_cons,
            total=total,
            weights={
                "w_performance": self.w_performance,
                "w_error": self.w_error,
                "w_stability": self.w_stability,
                "w_consistency": self.w_consistency,
            },
        )
        self.rewards.append(reward)
        self.prev_performance = performance_after
        return reward

    def baseline_relative_reward(
        self, current_performance: float) -> float:
        """Performance relative to baseline (not just current delta).

        Distinguishes: "was already good" vs "modification helped."
        """
        return current_performance - self.baseline_performance

    def reward_trajectory(self) -> List[Dict[str, Any]]:
        """Return reward trajectory for analysis."""
        return [r.to_dict() for r in self.rewards]

    def summary(self) -> Dict[str, Any]:
        """Summary statistics over all rewards."""
        if not self.rewards:
            return {"n_rewards": 0}
        totals = np.array([r.total for r in self.rewards])
        perfs = np.array([r.r_performance for r in self.rewards])
        return {
            "n_rewards": len(self.rewards),
            "mean_total": float(totals.mean()),
            "std_total": float(totals.std()),
            "mean_performance_delta": float(perfs.mean()),
            "positive_reward_fraction": float((totals > 0).mean()),
            "positive_performance_fraction": float((perfs > 0).mean()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {
                "w_performance": self.w_performance,
                "w_error": self.w_error,
                "w_stability": self.w_stability,
                "w_consistency": self.w_consistency,
            },
            "summary": self.summary(),
            "n_rewards": len(self.rewards),
        }


class PolicyCreditAssigner:
    """Assign credit to specific policy decisions based on subsequent outcomes.

    For each round t:
        Credit_t = f(Outcome_{t:t+k})

    where the credit signal determines how much the policy at round t
    contributed to the observed outcomes.

    This is distinct from ExperienceCreditAssigner (v0.5.3) which assigns
    temporal credit to *experiences*. PolicyCreditAssigner assigns credit
    to *policy decisions* (target, magnitude, direction choices).
    """

    def __init__(self, gamma: float = 0.95, credit_horizon: int = 5):
        self.gamma = gamma
        self.credit_horizon = credit_horizon
        self.decisions: Dict[int, Dict[str, Any]] = {}
        self.credits: Dict[int, float] = {}

    def record_decision(
        self,
        round_id: int,
        target: int,
        magnitude: float,
        reward: float,
    ) -> None:
        """Record a policy decision and its immediate reward."""
        self.decisions[round_id] = {
            "target": target,
            "magnitude": magnitude,
            "reward": reward,
        }

    def compute_credit(self, round_id: int,
                       future_rewards: Dict[int, float]) -> float:
        """Compute credit for the decision at round_id.

        G_t = Σ_{j=0}^{k-1} γ^j * r_{t+j}
        """
        credit = 0.0
        for j in range(self.credit_horizon):
            r = future_rewards.get(round_id + j, 0.0)
            credit += (self.gamma ** j) * r
        self.credits[round_id] = credit
        return credit

    def compute_all_credits(self,
                            round_rewards: Dict[int, float]) -> Dict[int, float]:
        """Compute credits for all recorded decisions."""
        for rid in self.decisions:
            self.compute_credit(rid, round_rewards)
        return self.credits.copy()

    def credit_by_target(self) -> Dict[int, Dict[str, float]]:
        """Mean credit grouped by target group."""
        by_target: Dict[int, List[float]] = {}
        for rid, credit in self.credits.items():
            target = self.decisions[rid]["target"]
            by_target.setdefault(target, []).append(credit)
        result = {}
        for t, vals in by_target.items():
            arr = np.array(vals)
            result[t] = {"mean": float(arr.mean()), "std": float(arr.std()),
                         "n": len(vals)}
        return result

    def summary(self) -> Dict[str, Any]:
        if not self.credits:
            return {"n_credits": 0}
        vals = np.array(list(self.credits.values()))
        return {
            "n_credits": len(self.credits),
            "mean_credit": float(vals.mean()),
            "std_credit": float(vals.std()),
            "credit_by_target": self.credit_by_target(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "credit_horizon": self.credit_horizon,
            "summary": self.summary(),
        }


class OutcomeDirectedPolicyLearner:
    """Complete outcome-directed policy learning pipeline.

    Integrates:
        1. ModificationRewardModel → computes R_t
        2. PolicyCreditAssigner → assigns credit to policy decisions
        3. Reward-conditioned policy update → updates policy using R_t

    This replaces v0.5.3's outcome-agnostic losses with explicit
    outcome-directed credit assignment.
    """

    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        credit_gamma: float = 0.95,
        credit_horizon: int = 5,
        reward_baseline_subtraction: bool = True,
    ):
        rw = reward_weights or {}
        self.reward_model = ModificationRewardModel(
            w_performance=rw.get("w_performance", 0.4),
            w_error=rw.get("w_error", 0.3),
            w_stability=rw.get("w_stability", 0.2),
            w_consistency=rw.get("w_consistency", 0.1),
        )
        self.credit_assigner = PolicyCreditAssigner(
            gamma=credit_gamma,
            credit_horizon=credit_horizon,
        )
        self.baseline_subtraction = reward_baseline_subtraction

        # track all decisions and outcomes
        self.round_rewards: Dict[int, float] = {}
        self.history: List[Dict[str, Any]] = []

    def step(
        self,
        round_id: int,
        performance_before: float,
        performance_after: float,
        param_norm_before: float,
        param_norm_after: float,
        error_before: float = 0.0,
        error_after: float = 0.0,
        target: int = 0,
        magnitude: float = 0.0,
    ) -> ModificationReward:
        """Process one round: compute reward, assign credit.

        Returns:
            ModificationReward for this round.
        """
        # compute reward
        reward = self.reward_model.compute_reward(
            round_id=round_id,
            performance_before=performance_before,
            performance_after=performance_after,
            param_norm_before=param_norm_before,
            param_norm_after=param_norm_after,
            error_before=error_before,
            error_after=error_after,
        )

        # record decision and reward
        self.credit_assigner.record_decision(
            round_id=round_id,
            target=target,
            magnitude=magnitude,
            reward=reward.total,
        )
        self.round_rewards[round_id] = reward.total

        # compute credits periodically
        if round_id % 5 == 0 and round_id > 0:
            self.credit_assigner.compute_all_credits(self.round_rewards)

        self.history.append({
            "round_id": round_id,
            "reward": reward.to_dict(),
            "target": target,
            "magnitude": magnitude,
        })

        return reward

    def get_credit(self, round_id: int) -> float:
        """Get the credit for a specific round's decision."""
        return self.credit_assigner.credits.get(round_id, 0.0)

    def summary(self) -> Dict[str, Any]:
        return {
            "reward_model": self.reward_model.summary(),
            "credit_assigner": self.credit_assigner.summary(),
            "n_rounds": len(self.history),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_model": self.reward_model.to_dict(),
            "credit_assigner": self.credit_assigner.to_dict(),
            "n_rounds": len(self.history),
        }
