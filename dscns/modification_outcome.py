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
