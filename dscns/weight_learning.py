"""Weight learning module (v0.5.2) — outcome-conditioned modification weight.

Learns the modification weight w_t ∈ [w_min, w_max] from the current state,
error history and retrieved memory, and enforces the ordering constraint

    w_success > w_failure

through a margin ranking loss (task spec §9-10).

Closed loop:

    (z_state, z_error, z_memory) -> WeightLearner -> w_t ∈ [w_min, w_max]
    w_t -> modification -> outcome ->
        OutcomeConditionedWeight.update_from_outcome
        (success -> confidence up, failure -> caution down,
         recovery -> w_correction up, repeated similar failure -> down more)

Components:

  * WeightLearner          -- nn.Module: state+error+memory -> weight scalar
  * WeightRankingLoss      -- L_ranking = max(0, margin - w_success + w_failure),
                              online (per-round) and batch variants
  * OutcomeConditionedWeight -- heuristic outcome-history modulation of w_t,
                              clamped to [w_min, w_max]

This is a model-side, torch-only component: no prompts, no code generation,
no exec() — the weight is produced by forward propagation only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------- #
# fixed weight range (task spec §9-10: w_min = 0.02, w_max = 1.0)        #
# ---------------------------------------------------------------------- #
W_MIN = 0.02
W_MAX = 1.0

# input dims: z_state (256) + z_error (32) + z_memory (32) = 320
DEFAULT_STATE_DIM = 256
DEFAULT_ERROR_DIM = 32
DEFAULT_MEMORY_DIM = 32
WEIGHT_INPUT_DIM = DEFAULT_STATE_DIM + DEFAULT_ERROR_DIM + DEFAULT_MEMORY_DIM

# default ranking margin (task spec: margin = 0.1)
DEFAULT_MARGIN = 0.1


class WeightLearner(nn.Module):
    """Learn the modification weight from state, error history and memory.

    Architecture (task spec §9-10):
        input  = [z_state (256); z_error (32); z_memory (32)]  -> 320 dims
        hidden = 64 dims with ReLU
        output = sigmoid -> [w_min, w_max]

    The final head is initialized so that a zero-pre-activation network
    produces a mid-range weight (≈ (w_min + w_max) / 2), keeping early
    modifications conservative.
    """

    def __init__(self, state_dim: int = DEFAULT_STATE_DIM,
                 error_dim: int = DEFAULT_ERROR_DIM,
                 memory_dim: int = DEFAULT_MEMORY_DIM,
                 hidden: int = 64,
                 w_min: float = W_MIN, w_max: float = W_MAX):
        super().__init__()
        self.state_dim = state_dim
        self.error_dim = error_dim
        self.memory_dim = memory_dim
        self.w_min = float(w_min)
        self.w_max = float(w_max)

        input_dim = state_dim + error_dim + memory_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_mid_range()

    # ------------------------------------------------------------------ #
    def _init_mid_range(self) -> None:
        """Bias the final linear so that the raw logit maps to mid-range."""
        with torch.no_grad():
            mid = (self.w_min + self.w_max) / 2.0
            frac = (mid - self.w_min) / max(self.w_max - self.w_min, 1e-8)
            frac = float(np.clip(frac, 1e-4, 1.0 - 1e-4))
            logit = float(np.log(frac / (1.0 - frac)))
            self.net[-1].bias.fill_(logit)

    # ------------------------------------------------------------------ #
    def forward(self, state: Any, error: Any, memory: Any) -> torch.Tensor:
        """Compute w = w_min + sigmoid(net([state; error; memory])) * (w_max - w_min).

        Accepts both single samples (dim == 1, returns a scalar-shaped
        tensor) and batches (B, dim, returns (B, 1)).
        """
        state = self._as_tensor(state)
        error = self._as_tensor(error)
        memory = self._as_tensor(memory)

        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)
            error = error.unsqueeze(0)
            memory = memory.unsqueeze(0)

        x = torch.cat([state, error, memory], dim=-1)      # (B, 320)
        logit = self.net(x)                                # (B, 1)
        weight = self.w_min + torch.sigmoid(logit) * (self.w_max - self.w_min)
        return weight.squeeze(0) if was_1d else weight

    # ------------------------------------------------------------------ #
    def ranking_loss(self, w_success: Any, w_failure: Any,
                     margin: float = DEFAULT_MARGIN) -> torch.Tensor:
        """L_ranking = max(0, margin - w_success + w_failure).

        Zero when the constraint w_success - w_failure >= margin holds;
        positive (a gradient signal) when the ordering is violated.
        """
        ws = self._as_tensor(w_success)
        wf = self._as_tensor(w_failure)
        return torch.clamp(margin - (ws - wf), min=0.0).mean()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_tensor(v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.float()
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            return torch.stack([x.float() for x in v])
        return torch.as_tensor(np.asarray(v, dtype=np.float32),
                               dtype=torch.float32)


class WeightRankingLoss(nn.Module):
    """Margin ranking loss that enforces w_success > w_failure (task spec §9).

    L_ranking = max(0, margin - w_success + w_failure)

    Supports both online (per-round, single pair) and batch training:
      - forward / loss_online : one (w_success, w_failure) pair or vectors
      - loss_batch            : mean over many pairs (same formula)
    """

    def __init__(self, margin: float = DEFAULT_MARGIN):
        super().__init__()
        self.margin = float(margin)

    # ------------------------------------------------------------------ #
    def forward(self, w_success: Any, w_failure: Any,
                margin: Optional[float] = None) -> torch.Tensor:
        """Mean margin ranking loss over the given (success, failure) pairs."""
        m = self.margin if margin is None else float(margin)
        ws = WeightLearner._as_tensor(w_success)
        wf = WeightLearner._as_tensor(w_failure)
        return torch.clamp(m - (ws - wf), min=0.0).mean()

    # ------------------------------------------------------------------ #
    def loss_online(self, w_success: Any, w_failure: Any,
                    margin: Optional[float] = None) -> torch.Tensor:
        """Online (per-round) ranking loss for a single outcome pair."""
        return self.forward(w_success, w_failure, margin)

    # ------------------------------------------------------------------ #
    def loss_batch(self, w_success_batch: Any, w_failure_batch: Any,
                   margin: Optional[float] = None) -> torch.Tensor:
        """Batch ranking loss: mean over all (success, failure) pairs."""
        return self.forward(w_success_batch, w_failure_batch, margin)


class OutcomeConditionedWeight:
    """Modulate the modification weight from the outcome history.

    Rules (task spec §9-10):
      - success  -> w increases  (confidence grows)
      - failure  -> w decreases  (caution grows)
      - recovery -> w_correction increases (correction strength up)
      - failure on a context similar to a past one -> w decreases more
        (repeated failure penalty)

    All adjustments are clamped to [w_min, w_max]; every update is recorded
    in ``history`` for analysis.
    """

    def __init__(self, w_min: float = W_MIN, w_max: float = W_MAX,
                 success_step: float = 0.1,
                 failure_step: float = 0.1,
                 recovery_boost: float = 0.15,
                 repeated_failure_penalty: float = 0.1,
                 similarity_threshold: float = 0.6):
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.success_step = float(success_step)
        self.failure_step = float(failure_step)
        self.recovery_boost = float(recovery_boost)
        self.repeated_failure_penalty = float(repeated_failure_penalty)
        self.similarity_threshold = float(similarity_threshold)

        self.w_correction = 0.0          # correction weight after recovery
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def update_from_outcome(self, current_weight: float, outcome: str,
                            similarity_to_past: float = 0.0) -> float:
        """Return the adjusted weight after applying the outcome rule.

        Args:
            current_weight: w_t before the update.
            outcome: "success" | "partial_success" | "failure" |
                     "catastrophic" | "recovery" | "neutral".
            similarity_to_past: similarity of the current context to past
                experiences (used to detect repeated similar failures).

        Returns:
            adjusted weight w_{t+1} ∈ [w_min, w_max].
        """
        w = float(current_weight)
        outcome = str(outcome).lower()
        similarity = float(similarity_to_past)

        if outcome in ("success", "partial_success"):
            w += self.success_step                                   # confidence
        elif outcome == "recovery":
            w += self.success_step                                   # confidence
            self.w_correction = min(
                self.w_max, self.w_correction + self.recovery_boost)  # w_corr up
        elif outcome in ("failure", "catastrophic"):
            w -= self.failure_step                                   # caution
            if similarity >= self.similarity_threshold:
                w -= self.repeated_failure_penalty                   # repeat down
        # neutral / unknown outcomes leave the weight unchanged

        w = float(np.clip(w, self.w_min, self.w_max))
        self.history.append({
            "outcome": outcome,
            "similarity_to_past": similarity,
            "w_before": float(current_weight),
            "w_after": w,
            "w_correction": self.w_correction,
        })
        return w

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear the correction weight and the update history."""
        self.w_correction = 0.0
        self.history = []


__all__ = [
    "W_MIN", "W_MAX", "DEFAULT_MARGIN", "WEIGHT_INPUT_DIM",
    "WeightLearner", "WeightRankingLoss", "OutcomeConditionedWeight",
]
