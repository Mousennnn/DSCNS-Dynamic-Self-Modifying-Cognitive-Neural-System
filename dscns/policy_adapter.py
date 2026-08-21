"""Policy Adapter (v0.5.3 / Phase 5.5).

THE key architectural change from v0.5.2:  Memory now directly conditions
the Modification Policy, not just the Correction signal.

v0.5.2 had:
    Memory → Correction → (applied next round)

v0.5.3 has:
    Experience → Policy Adapter → Modification Proposal

The Policy Adapter replaces the simple IntrinsicPlasticityModule's
generate_proposal() by adding an experience-conditioned layer:

    z_t = [state_t, error_t, memory_t, experience_value_t]
    π_t(Target, Magnitude, Direction) = PolicyAdapter(z_t)

This means that when similar experiences are retrieved from memory,
they directly influence what modification is proposed — not what
correction is applied afterward.

Components:
  PolicyAdapter        -- the core experience-conditioned policy
  PolicyState          -- snapshotable policy state
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyAdapter(nn.Module):
    """Experience-conditioned modification policy.

    Architecture:
        Input: [z_state(256), z_error(32), z_memory(32), z_exp_value(16)] = 336
        → Fusion(336→256) →
            → TargetHead(256→3)        — target group distribution
            → MagnitudeHead(256→1)     — magnitude in [m_min, m_max]
            → DirectionModulation(256→32) — additive mod to core z
            → CandidateScoreHead(256→K) — scores for K candidate strategies

    The output modifies the base plasticity module's proposal:
        target = argmax(TargetHead) with exploration
        magnitude = MagnitudeHead
        direction = base_direction + DirectionModulation
    """

    def __init__(self, state_dim: int = 256, error_dim: int = 32,
                 memory_dim: int = 32, value_dim: int = 16,
                 hidden_dim: int = 256, n_candidates: int = 4,
                 m_min: float = 0.02, m_max: float = 1.0,
                 n_target_groups: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.error_dim = error_dim
        self.memory_dim = memory_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.n_candidates = n_candidates
        self.m_min = m_min
        self.m_max = m_max
        self.n_target_groups = n_target_groups

        input_dim = state_dim + error_dim + memory_dim + value_dim

        # fusion backbone
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # target head: which parameter group to modify
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, n_target_groups),
        )

        # magnitude head: how much to modify [m_min, m_max]
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

        # direction modulation: additive offset to the base direction
        self.direction_mod = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 32),
        )

        # candidate scoring head: scores K candidate modification strategies
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, n_candidates),
        )

        # confidence head: how confident the policy is
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.Sigmoid(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        # statistics
        self._n_forward_calls = 0

    def forward(
        self,
        state_z: torch.Tensor,          # (B, 256) from plasticity core z
        error_z: torch.Tensor,           # (B, 32) from error encoder
        memory_z: torch.Tensor,          # (B, 32) from memory retriever
        exp_value_z: torch.Tensor,       # (B, 16) from experience value
    ) -> Dict[str, Any]:
        """Generate experience-conditioned modification policy.

        Returns dict with:
            target_logits: (B, n_targets) raw target distribution
            target_probs: (B, n_targets) softmax target distribution
            magnitude: (B, 1) in [m_min, m_max]
            direction_mod: (B, 32) additive direction modulation
            candidate_scores: (B, K) raw scores for K candidates
            confidence: (B, 1) policy confidence
            fused_z: (B, hidden_dim) for downstream use
        """
        B = state_z.size(0)
        device = state_z.device

        # ensure correct dimensions
        state_z = state_z[:, :self.state_dim] if state_z.size(-1) > self.state_dim else \
            F.pad(state_z, (0, self.state_dim - state_z.size(-1)))
        error_z = error_z[:, :self.error_dim] if error_z.size(-1) > self.error_dim else \
            F.pad(error_z, (0, self.error_dim - error_z.size(-1)))
        memory_z = memory_z[:, :self.memory_dim] if memory_z.size(-1) > self.memory_dim else \
            F.pad(memory_z, (0, self.memory_dim - memory_z.size(-1)))
        exp_value_z = exp_value_z[:, :self.value_dim] if exp_value_z.size(-1) > self.value_dim else \
            F.pad(exp_value_z, (0, self.value_dim - exp_value_z.size(-1)))

        x = torch.cat([state_z, error_z, memory_z, exp_value_z], dim=-1)
        fused = self.fusion(x)

        target_logits = self.target_head(fused)
        target_probs = F.softmax(target_logits, dim=-1)

        mag_raw = torch.sigmoid(self.magnitude_head(fused))
        magnitude = self.m_min + mag_raw * (self.m_max - self.m_min)

        dir_mod = self.direction_mod(fused)

        cand_scores = self.candidate_head(fused)
        confidence = self.confidence_head(fused)

        self._n_forward_calls += 1

        return {
            "target_logits": target_logits,
            "target_probs": target_probs,
            "magnitude": magnitude,
            "direction_mod": dir_mod,
            "candidate_scores": cand_scores,
            "confidence": confidence,
            "fused_z": fused,
        }

    def select_target(self, target_probs: torch.Tensor,
                      exploration_eps: float = 0.1,
                      exploration_min: float = 0.02,
                      ) -> torch.Tensor:
        """Select target with ε-greedy exploration.

        With probability ε: random target
        With probability 1-ε: argmax target

        Args:
            target_probs: (B, n_targets) probability distribution.
            exploration_eps: current exploration rate.
            exploration_min: minimum exploration rate.

        Returns:
            (B,) selected target indices.
        """
        B = target_probs.size(0)
        device = target_probs.device
        eps = max(exploration_eps, exploration_min)

        # greedy selection
        greedy = target_probs.argmax(dim=-1)  # (B,)

        # random selection
        random_targets = torch.randint(0, self.n_target_groups, (B,), device=device)

        # epsilon-greedy mask
        explore_mask = (torch.rand(B, device=device) < eps).long()
        selected = explore_mask * random_targets + (1 - explore_mask) * greedy
        return selected

    def get_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return list(self.parameters())


@dataclass
class PolicyState:
    """Snapshotable policy state for checkpointing."""
    target_distribution: Optional[Dict[int, float]] = None
    mean_magnitude: float = 0.0
    mean_confidence: float = 0.0
    exploration_rate: float = 0.1
    n_forward_calls: int = 0
    round_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_distribution": self.target_distribution,
            "mean_magnitude": self.mean_magnitude,
            "mean_confidence": self.mean_confidence,
            "exploration_rate": self.exploration_rate,
            "n_forward_calls": self.n_forward_calls,
            "round_id": self.round_id,
        }
