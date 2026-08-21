"""Modification outcome evaluation (P5.2, design report section 16-18).

Defines before/after scores, SUCCESS/FAILURE/RECOVERY classification,
and the causal evaluation framework.  Outcome is NOT parameter drift —
it is the actual behavioral consequence of a modification.

Thresholds are fixed BEFORE the experiment (no post-hoc tuning).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np


@dataclass
class ModificationOutcome:
    """Complete outcome record for one modification event."""
    round_id: int = 0
    # scores (frozen probe set, task performance, loss)
    score_before: float = 0.0         # probe metric at θ_t
    score_after: float = 0.0          # probe metric at θ'_t
    loss_before: float = 0.0
    loss_after: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    # deltas
    delta_score: float = 0.0          # score_after - score_before
    delta_loss: float = 0.0           # loss_after - loss_before
    delta_entropy: float = 0.0
    # modification details
    delta_norm: float = 0.0           # ||Δθ_effective||
    weight: float = 0.0              # w_t (applied weight)
    target_group: int = -1
    # correction (None if not a correction round)
    correction_applied: bool = False
    correction_norm: float = 0.0
    # classification
    outcome: str = "neutral"          # success / failure / partial_success / catastrophic
    category: str = "success"         # success / failure / recovery
    # rollback
    rolled_back: bool = False
    rolled_back_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                d[k] = v
        return d


class OutcomeEvaluator:
    """Classifies modification outcomes using fixed thresholds.

    Thresholds are set at init time and MUST NOT be changed during the experiment.
    """

    def __init__(self,
                 success_threshold: float = 0.0001,
                 failure_threshold: float = -0.0001,
                 catastrophic_entropy: float = 0.1,
                 catastrophic_param_norm: float = 1000.0,
                 catastrophic_nan: bool = True):
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.catastrophic_entropy = catastrophic_entropy
        self.catastrophic_param_norm = catastrophic_param_norm
        self.catastrophic_nan = catastrophic_nan

    def evaluate(self, score_before: float, score_after: float,
                 loss_before: float = 0.0, loss_after: float = 0.0,
                 entropy_before: float = 4.0, entropy_after: float = 4.0,
                 param_norm: float = 10.0, has_nan: bool = False,
                 delta_score: float = 0.0) -> Dict[str, Any]:
        """Classify the modification outcome.

        Uses the PROBE set score (not task loss) as the primary signal.
        """
        ds = score_after - score_before if abs(delta_score) < 0.0001 else delta_score
        success_a = ds > self.success_threshold     # any improvement
        success_b = ds > self.success_threshold * 5  # clear improvement
        failure_c = ds < self.failure_threshold      # clear degradation
        catastrophic = (has_nan or
                        entropy_after < self.catastrophic_entropy or
                        param_norm > self.catastrophic_param_norm)
        if catastrophic:
            outcome, category = "catastrophic", "failure"
        elif failure_c:
            outcome, category = "failure", "failure"
        elif success_b:
            outcome, category = "success", "success"
        elif success_a:
            outcome, category = "partial_success", "success"
        else:
            outcome, category = "neutral", "neutral"

        return {
            "delta_score": ds,
            "success_a": success_a,
            "success_b": success_b,
            "failure_c": failure_c,
            "catastrophic": catastrophic,
            "outcome": outcome,
            "category": category,
        }

    def classify_recovery(self, prev_outcome: str, curr_outcome: str,
                          prev_score: float, curr_score: float) -> str:
        """Classify if current round is a RECOVERY from previous failure."""
        if prev_outcome == "failure" and curr_outcome in ("success", "partial_success"):
            if curr_score > prev_score:
                return "recovery"
        return curr_outcome


class FailureInjector:
    """Deliberately induces failures at specified rounds (Exp B, section 27).

    At injection rounds, the modification is forced to use max magnitude
    and/or a sensitive target, ensuring probe degradation.
    """

    def __init__(self, injection_rounds: List[int],
                 injection_magnitude: float = 1.0,
                 injection_target: Optional[int] = None,
                 injection_alpha: float = 0.05):
        self.injection_rounds = set(injection_rounds)
        self.injection_magnitude = injection_magnitude
        self.injection_target = injection_target
        self.injection_alpha = injection_alpha
        self.injections_done = 0

    def should_inject(self, round_id: int) -> bool:
        return round_id in self.injection_rounds

    def get_injection_params(self) -> Dict[str, Any]:
        self.injections_done += 1
        return {
            "magnitude": self.injection_magnitude,
            "target_group": self.injection_target,  # None = random
            "alpha": self.injection_alpha,
            "injected": True,
        }


# ====================================================================== #
# v0.5.1: Separated Recovery Metrics (task spec §16)                    #
# ====================================================================== #

@dataclass
class RecoveryMetrics:
    """Separated metrics for correction effectiveness (task spec §16).

    Three metrics that MUST NOT be merged:
      CAR: system attempts correction
      SRR: correction actually succeeds
      RE: how much of the original loss is recovered
    """
    correction_application_rate: float = 0.0   # CAR = N_corrections / N_failures
    successful_recovery_rate: float = 0.0      # SRR = N_successful_recovery / N_failures
    recovery_efficiency: float = 0.0           # RE = (P_after_corr - P_after_fail) / (P_before - P_after_fail + eps)

    # component counts
    total_failures: int = 0
    total_corrections_applied: int = 0
    total_successful_recoveries: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "CAR": self.correction_application_rate,
            "SRR": self.successful_recovery_rate,
            "RE": self.recovery_efficiency,
            "total_failures": self.total_failures,
            "total_corrections_applied": self.total_corrections_applied,
            "total_successful_recoveries": self.total_successful_recoveries,
        }


class V051OutcomeEvaluator:
    """v0.5.1 outcome evaluator with separated recovery metrics.

    Extends OutcomeEvaluator with:
      - Correction Application Rate (CAR)
      - Successful Recovery Rate (SRR)
      - Recovery Efficiency (RE)
      - Similar failure tracking
      - Weight adaptation tracking

    Thresholds are fixed BEFORE the experiment (no post-hoc tuning).
    """

    def __init__(self,
                 success_threshold: float = 0.0001,
                 failure_threshold: float = -0.0001,
                 recovery_threshold: float = 0.0001,
                 catastrophic_entropy: float = 0.1,
                 catastrophic_param_norm: float = 1000.0):
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.catastrophic_entropy = catastrophic_entropy
        self.catastrophic_param_norm = catastrophic_param_norm

    def evaluate(self, score_before: float, score_after: float,
                 loss_before: float = 0.0, loss_after: float = 0.0,
                 entropy_before: float = 4.0, entropy_after: float = 4.0,
                 param_norm: float = 10.0, has_nan: bool = False,
                 delta_score: float = 0.0) -> Dict[str, Any]:
        """Classify modification outcome with fixed thresholds."""
        ds = score_after - score_before if abs(delta_score) < 0.0001 else delta_score
        catastrophic = (has_nan or
                        entropy_after < self.catastrophic_entropy or
                        param_norm > self.catastrophic_param_norm)
        if catastrophic:
            outcome, category = "catastrophic", "failure"
        elif ds < self.failure_threshold:
            outcome, category = "failure", "failure"
        elif ds > self.success_threshold * 5:
            outcome, category = "success", "success"
        elif ds > self.success_threshold:
            outcome, category = "partial_success", "success"
        else:
            outcome, category = "neutral", "neutral"
        return {
            "delta_score": ds,
            "outcome": outcome,
            "category": category,
            "catastrophic": catastrophic,
        }

    def compute_recovery_metrics(
        self,
        outcomes: List[Dict[str, Any]],
    ) -> RecoveryMetrics:
        """v0.5.1: compute CAR, SRR, RE from a list of round outcomes.

        Each outcome dict must contain:
          - category: "success"/"failure"/"recovery"
          - correction_applied: bool
          - score_before_modification: float
          - score_after_modification: float
          - score_after_correction: float (if correction applied)
        """
        metrics = RecoveryMetrics()
        for ev in outcomes:
            if ev.get("category") == "failure":
                metrics.total_failures += 1
                if ev.get("correction_applied", False):
                    metrics.total_corrections_applied += 1
                    # check if recovery actually happened
                    score_after_fail = ev.get("score_after_modification", 0.0)
                    score_after_corr = ev.get("score_after_correction", score_after_fail)
                    score_before = ev.get("score_before_modification", 0.0)
                    if score_after_corr > score_after_fail + self.recovery_threshold:
                        metrics.total_successful_recoveries += 1

        metrics.correction_application_rate = (
            metrics.total_corrections_applied / max(metrics.total_failures, 1))
        metrics.successful_recovery_rate = (
            metrics.total_successful_recoveries / max(metrics.total_failures, 1))

        # RE averaged over failures that had corrections
        re_values = []
        for ev in outcomes:
            if ev.get("category") == "failure" and ev.get("correction_applied", False):
                score_before = ev.get("score_before_modification", 0.0)
                score_after_fail = ev.get("score_after_modification", 0.0)
                score_after_corr = ev.get("score_after_correction", score_after_fail)
                denom = abs(score_before - score_after_fail) + 1e-8
                re_values.append((score_after_corr - score_after_fail) / denom)
        if re_values:
            metrics.recovery_efficiency = float(np.mean(re_values))

        return metrics


class NaturalFailureDetector:
    """v0.5.1: detect natural failures (task spec §19-20).

    A 'natural failure' is a failure NOT caused by injection —
    the model's own modification led to performance degradation.

    This is the most important experiment for proving the model
    learns from its OWN errors.
    """

    def __init__(self, failure_threshold: float = -0.0001):
        self.failure_threshold = failure_threshold
        self.natural_failures: List[Dict[str, Any]] = []
        self.total_natural_rounds = 0
        self.total_natural_failures = 0

    def record_round(self, round_id: int, delta_score: float,
                     injected: bool = False, **kwargs) -> Dict[str, Any]:
        """Record a round's outcome and classify if it's a natural failure."""
        is_natural_failure = (not injected and delta_score < self.failure_threshold)
        if is_natural_failure:
            self.total_natural_failures += 1
            self.natural_failures.append({
                "round_id": round_id,
                "delta_score": delta_score,
                **kwargs,
            })
        self.total_natural_rounds += 1
        return {
            "is_natural_failure": is_natural_failure,
            "natural_failure_count": self.total_natural_failures,
        }

    @property
    def natural_failure_rate(self) -> float:
        if self.total_natural_rounds == 0:
            return 0.0
        return self.total_natural_failures / self.total_natural_rounds
