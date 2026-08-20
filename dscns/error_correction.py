"""Error-conditioned self-correction mechanism (P5.1).

Encodes modification outcomes into an error representation that conditions
the next self-modification decision.  This is the model-side learning
mechanism (not a rule-based controller): the error representation enters
P_phi as an additional input, and P_phi learns to produce different
modification behaviors conditioned on past outcomes.

Key components:
  ErrorState       -- per-round structured error representation
  ErrorEncoder     -- MLP that maps ErrorState -> error embedding
  OutcomeEvaluator -- determines success / failure / recovery
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn


# ---- outcome levels (report section 13) ----
LEVEL_A_THRESHOLD = 0.001   # local success: task delta > threshold
LEVEL_B_THRESHOLD = 0.001   # generalization: probe delta > threshold
LEVEL_C_THRESHOLD = -0.001  # regression: probe delta < -threshold
SOFT_DEGRADATION = -0.001
SIGNIFICANT_DEGRADATION = -0.05


@dataclass
class ErrorState:
    """Structured error representation from one modification round."""
    task_delta: float = 0.0         # perf_after - perf_before
    probe_delta: float = 0.0        # probe_out_drift_after - before
    forgetting_delta: float = 0.0
    logit_delta: float = 0.0        # change in mean |logits diff|
    entropy_delta: float = 0.0      # entropy_after - entropy_before
    parameter_drift: float = 0.0    # ||Δθ||
    prev_target: int = -1           # previous target group (-1 = none)
    prev_magnitude: float = 0.0     # previous magnitude

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.task_delta, self.probe_delta, self.forgetting_delta,
            self.logit_delta, self.entropy_delta, self.parameter_drift,
            float(self.prev_target) / 3.0, self.prev_magnitude,
        ], dtype=torch.float32)

    @staticmethod
    def dim() -> int:
        return 8


@dataclass
class ModificationProposal:
    """Complete proposal from P_phi (report section 8)."""
    delta_W_A: Any = None        # (768, 16) consensus delta
    delta_W_B: Any = None        # (16, 768)
    magnitude: float = 0.05      # m_t ∈ [m_min, m_max]
    target_group: int = 0        # 0=attn_A, 1=attn_B, 2=mlp_B
    confidence: float = 0.0      # max prob from target head
    modulation_strength: float = 0.0


class ErrorEncoder(nn.Module):
    """Maps ErrorState -> error embedding for P_phi conditioning.

    Input: 8-dim error vector (see ErrorState.to_tensor).
    Output: error_dim-dim embedding.
    """
    def __init__(self, error_dim: int = 32):
        super().__init__()
        in_dim = ErrorState.dim()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, error_dim),
        )

    def forward(self, error_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(error_tensor)


class OutcomeEvaluator:
    """Determines modification outcome categories.

    Levels:
      A -- local success (task improves)
      B -- generalization success (probe improves)
      C -- regression (probe degrades significantly)
    """
    def __init__(self, success_thresh: float = LEVEL_B_THRESHOLD,
                 failure_thresh: float = LEVEL_C_THRESHOLD,
                 catastrophic_check=None):
        self.success_thresh = success_thresh
        self.failure_thresh = failure_thresh

    def evaluate(self, perf_before: float, perf_after: float,
                 probe_before: float, probe_after: float,
                 entropy: float = 4.0, param_norm: float = 10.0,
                 has_nan: bool = False) -> Dict[str, Any]:
        task_delta = perf_after - perf_before
        probe_delta = probe_after - probe_before
        success_a = task_delta > LEVEL_A_THRESHOLD
        success_b = probe_delta > self.success_thresh
        failure_c = probe_delta < self.failure_thresh
        catastrophic = has_nan or entropy < 0.1 or param_norm > 1000
        if catastrophic:
            outcome = "catastrophic"
        elif failure_c:
            outcome = "failure"
        elif success_b:
            outcome = "success"
        elif success_a:
            outcome = "partial_success"
        else:
            outcome = "neutral"
        return {
            "task_delta": task_delta,
            "probe_delta": probe_delta,
            "success_a": success_a,
            "success_b": success_b,
            "failure_c": failure_c,
            "catastrophic": catastrophic,
            "outcome": outcome,
        }
