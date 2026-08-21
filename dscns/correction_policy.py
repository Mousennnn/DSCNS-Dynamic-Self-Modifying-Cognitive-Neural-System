"""Memory-conditioned correction policy (v0.5.1, task spec §§7-8, 11-12).

Replaces the simple CorrectionGenerator with a policy that:
  1. Takes ErrorState + retrieved memory + previous proposal as input
  2. Outputs a correction signal (NOT simply -Δθ)
  3. Conditions future modification weight, target, and direction
  4. Supports C0-C5 ablation modes

Key principle: Correction ≠ Rollback. The policy learns to blend:
  - Reversal component: α_t * (-Δθ_t)
  - Learned correction: (1-α_t) * C_learned(E_t, M_t, context_t)
  - Memory-conditioned adjustment: f(M_t) → weight/target shift

C0-C5 Ablation Modes:
  C0: No correction (no-op)
  C1: Fixed rollback to pre-modification state
  C2: Pure reversal (-Δθ)
  C3: Fixed learned correction (no memory)
  C4: Error-conditioned correction (no memory)
  C5: Error + Memory conditioned correction (full model)

Architecture:
  ErrorEncoder(8→32) + MemoryPolicyEncoder(32→32) + ProposalEncoder(→32)
  → Fusion(96→64) →
    → CorrectionHead(64→768*16 + 16*768) for correction signal
    → AlphaHead(64→1) for reversal weight α ∈ [0,1]
    → WeightAdjustHead(64→1) for next-round weight adjustment
    → TargetAdjustHead(64→3) for next-round target distribution
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---- Correction modes ----

CORRECTION_MODES = ["none", "rollback", "reversal", "learned",
                    "error_conditioned", "memory_conditioned"]
MODE_NONE = "none"              # C0
MODE_ROLLBACK = "rollback"      # C1
MODE_REVERSAL = "reversal"      # C2
MODE_LEARNED = "learned"        # C3
MODE_ERROR_COND = "error_conditioned"  # C4
MODE_MEMORY_COND = "memory_conditioned"  # C5 (full)


@dataclass
class CorrectionOutput:
    """Output of the correction policy."""
    # correction signal (applied in next round)
    correction_W_A: Any = None        # (768, 16) correction delta
    correction_W_B: Any = None        # (16, 768) correction delta
    correction_norm: float = 0.0
    correction_strength: float = 0.0

    # reversal blending
    reversal_alpha: float = 0.0       # α_t ∈ [0,1]

    # policy adjustments for next round
    weight_adjustment: float = 0.0    # additive adjustment to w_{t+1}
    target_logits: Any = None         # (3,) target distribution adjustment
    confidence: float = 0.0

    # mode used
    mode: str = "memory_conditioned"

    # memory attention (for analysis)
    memory_attention: Any = None      # (k,) attention weights over retrieved


class CorrectionPolicy(nn.Module):
    """Memory-conditioned correction policy (C5 = full model).

    Produces correction signals and policy adjustments conditioned on:
      - ErrorState from the failed modification
      - Retrieved memory episodes
      - Previous modification proposal
      - Current context embedding
    """

    def __init__(self, error_dim: int = 8, memory_dim: int = 32,
                 core_dim: int = 256, proposal_dim: int = 16,
                 hidden_dim: int = 64, correction_dim: int = 128):
        super().__init__()
        self.error_dim = error_dim
        self.memory_dim = memory_dim
        self.core_dim = core_dim

        # ---- encoders ----
        # error encoder: ErrorState → z_error
        self.error_encoder = nn.Sequential(
            nn.Linear(error_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )

        # memory encoder: z_memory summary → z_mem
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )

        # proposal encoder: prev proposal info → z_prop
        # input: [dA_norm(1), weight(1), target(1)]
        self.proposal_encoder = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16),
        )

        # context encoder: core_z → z_ctx (reuse plasticity's z)
        self.context_encoder = nn.Sequential(
            nn.Linear(core_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )

        # ---- fusion: [z_error, z_mem, z_prop, z_ctx] → z_fused (96→64) ----
        fusion_dim = 32 + 32 + 16 + 32  # = 112
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- output heads ----
        # correction signal: ΔW correction for next round
        self.corr_A_head = nn.Linear(hidden_dim, 768 * 16)
        self.corr_B_head = nn.Linear(hidden_dim, 16 * 768)

        # alpha head: reversal weight α ∈ [0,1]
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.Sigmoid(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        # strength head: correction strength ∈ [0,1]
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.Sigmoid(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        # weight adjustment head: additive shift for next w
        self.weight_adj_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Tanh(),  # ∈ [-1, 1]
        )

        # target adjustment head: shift in target distribution
        self.target_adj_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(
        self,
        error_state: torch.Tensor,       # (B, error_dim)
        memory_z: torch.Tensor,          # (B, memory_dim) or (B, k, memory_dim)
        prev_proposal_features: torch.Tensor,  # (B, 7)
        core_z: torch.Tensor,            # (B, core_dim)
        mode: str = "memory_conditioned",
    ) -> Dict[str, Any]:
        """Generate correction signal and policy adjustments.

        Args:
            error_state: (B, 8) error representation
            memory_z: memory representations (summarized or batch)
            prev_proposal_features: (B, 7) previous proposal info
            core_z: (B, core_dim) current context from plasticity
            mode: correction ablation mode

        Returns:
            dict with correction_W_A, correction_W_B, alpha, strength,
                 weight_adjustment, target_logits, memory_attention
        """
        B = error_state.size(0)
        device = error_state.device

        # encode inputs
        z_error = self.error_encoder(error_state)     # (B, 32)

        # handle memory: (B, k, mem_dim) → (B, mem_dim) via pooling
        memory_attention = None
        if memory_z is not None:
            if memory_z.dim() == 3:
                # multi-episode: mean pooling (simple and stable)
                z_mem = memory_z.mean(dim=1)  # (B, mem_dim)
            elif memory_z.dim() == 2:
                z_mem = memory_z
            else:
                z_mem = torch.zeros(B, self.memory_dim, device=device)
        else:
            z_mem = torch.zeros(B, self.memory_dim, device=device)

        z_mem = self.memory_encoder(z_mem[:, :self.memory_dim])  # (B, 32)
        z_prop = self.proposal_encoder(prev_proposal_features)   # (B, 16)
        z_ctx = self.context_encoder(core_z)                     # (B, 32)

        # fusion
        z_fused = self.fusion(torch.cat([z_error, z_mem, z_prop, z_ctx], dim=-1))

        # mode-dependent output
        if mode == MODE_NONE:
            # C0: no correction
            return self._zero_output(B, device)
        elif mode == MODE_ROLLBACK:
            # C1: rollback (not implemented in model — handled by controller)
            return self._zero_output(B, device, mode=MODE_ROLLBACK)
        elif mode == MODE_REVERSAL:
            # C2: pure reversal — controller applies -Δθ
            return self._zero_output(B, device, mode=MODE_REVERSAL)

        # C3/C4/C5: learned correction with varying inputs
        if mode == MODE_LEARNED:
            # C3: ignore memory
            z_fused_no_mem = self.fusion(torch.cat([
                z_error, torch.zeros_like(z_mem), z_prop, z_ctx
            ], dim=-1))
            z_out = z_fused_no_mem
        elif mode == MODE_ERROR_COND:
            # C4: ignore memory
            z_out = z_fused  # memory already zeroed in input
        else:
            # C5: full memory-conditioned
            z_out = z_fused

        # generate correction
        corr_A = self.corr_A_head(z_out).view(B, 768, 16) * 0.01  # conservative init
        corr_B = self.corr_B_head(z_out).view(B, 16, 768) * 0.01
        alpha = self.alpha_head(z_out).squeeze(-1)                  # (B,)
        strength = self.strength_head(z_out).squeeze(-1)            # (B,)
        weight_adj = self.weight_adj_head(z_out).squeeze(-1)       # (B,) ∈ [-1,1]
        target_logits = self.target_adj_head(z_out)                 # (B, 3)

        return {
            "correction_W_A": corr_A.mean(dim=0),     # (768, 16)
            "correction_W_B": corr_B.mean(dim=0),     # (16, 768)
            "correction_norm": float(corr_A.norm() + corr_B.norm()),
            "correction_strength": float(strength.mean()),
            "reversal_alpha": float(alpha.mean()),
            "weight_adjustment": float(weight_adj.mean()),
            "target_logits": target_logits.mean(dim=0).detach(),
            "confidence": float(target_logits.softmax(-1).max()),
            "mode": mode,
            "memory_attention": memory_attention,
        }

    def _zero_output(self, B, device, mode="none"):
        return {
            "correction_W_A": torch.zeros(768, 16, device=device),
            "correction_W_B": torch.zeros(16, 768, device=device),
            "correction_norm": 0.0,
            "correction_strength": 0.0,
            "reversal_alpha": 0.0,
            "weight_adjustment": 0.0,
            "target_logits": torch.zeros(3, device=device),
            "confidence": 0.0,
            "mode": mode,
            "memory_attention": None,
        }


class CorrectionPolicyWithMemory(nn.Module):
    """Wrapper that adds memory retrieval to the CorrectionPolicy.

    This is the top-level module that:
      1. Encodes current episode
      2. Retrieves similar episodes from memory
      3. Encodes retrieved episodes
      4. Feeds everything to CorrectionPolicy
      5. Returns correction + policy adjustments
    """

    def __init__(self, error_dim: int = 8, memory_dim: int = 32,
                 core_dim: int = 256, hidden_dim: int = 64,
                 correction_dim: int = 128,
                 memory_top_k: int = 8,
                 lambda_context: float = 0.3,
                 lambda_proposal: float = 0.3,
                 lambda_error: float = 0.2,
                 lambda_target: float = 0.2):
        super().__init__()

        self.policy = CorrectionPolicy(
            error_dim=error_dim,
            memory_dim=memory_dim,
            core_dim=core_dim,
            hidden_dim=hidden_dim,
            correction_dim=correction_dim,
        )

        self.memory_encoder = None  # lazy init (needs MemoryEncoder from memory_encoder.py)
        self.memory_top_k = memory_top_k

        from .memory_encoder import MultiSimilarityRetriever, MemoryPolicyEncoder
        self.retriever = MultiSimilarityRetriever(
            top_k=memory_top_k,
            lambda_context=lambda_context,
            lambda_proposal=lambda_proposal,
            lambda_error=lambda_error,
            lambda_target=lambda_target,
        )
        self.memory_policy_encoder = MemoryPolicyEncoder(
            memory_dim=memory_dim, hidden_dim=hidden_dim,
        )

    def forward(
        self,
        error_state_tensor: torch.Tensor,  # (B, error_dim)
        core_z: torch.Tensor,              # (B, core_dim) current context
        prev_delta_A: torch.Tensor,        # (768, 16)
        prev_delta_B: torch.Tensor,        # (16, 768)
        prev_weight: float,
        prev_target: int,
        episodes: List,                     # list of ModificationEpisode
        mode: str = "memory_conditioned",
        prev_error: Any = None,
    ) -> Dict[str, Any]:
        """Full correction pipeline with memory retrieval.

        Returns correction output dict (same format as CorrectionPolicy).
        """
        B = error_state_tensor.size(0)
        device = error_state_tensor.device

        # 1. retrieve similar episodes
        retrieved = self.retriever.retrieve(
            query_context=core_z.mean(dim=0) if core_z.dim() > 1 else core_z,
            query_proposal=torch.tensor(
                [prev_weight, float(prev_target)], device=device),
            query_error=error_state_tensor.mean(dim=0) if error_state_tensor.dim() > 1
            else error_state_tensor,
            query_target=prev_target,
            episodes=episodes,
        )

        # 2. encode retrieved episodes into memory representation
        if retrieved:
            mem_features = []
            for sim, ep in retrieved:
                # handle both ModificationEpisode and EpisodicModificationRecord
                if hasattr(ep, "to_feature_vector"):
                    feat = ep.to_feature_vector(device)
                else:
                    # build feature vector from EpisodicModificationRecord attributes
                    target_oh = torch.zeros(3, device=device)
                    target_oh[min(getattr(ep, "target_group", 0), 2)] = 1.0
                    err = torch.zeros(8, device=device)
                    if hasattr(ep, "error_state") and ep.error_state is not None:
                        if hasattr(ep.error_state, "to_tensor"):
                            err = ep.error_state.to_tensor().to(device)
                    outcome_oh = torch.zeros(3, device=device)
                    cat = getattr(ep, "category", "success")
                    if cat == "success":
                        outcome_oh[0] = 1.0
                    elif cat == "failure":
                        outcome_oh[1] = 1.0
                    elif cat == "recovery":
                        outcome_oh[2] = 1.0
                    feat = torch.cat([
                        torch.tensor([getattr(ep, "delta_norm", 0.0),
                                      getattr(ep, "magnitude", 0.0)], device=device),
                        target_oh,
                        torch.tensor([getattr(ep, "probe_delta", 0.0)], device=device),
                        err,
                        outcome_oh,
                        torch.tensor([getattr(ep, "correction_norm", 0.0)], device=device),
                    ])
                mem_features.append(feat)
            mem_z = torch.stack(mem_features)  # (k, feat_dim)

            # pad to memory_dim if needed
            if mem_z.size(-1) < 32:
                pad = torch.zeros(mem_z.size(0), 32 - mem_z.size(-1), device=device)
                mem_z = torch.cat([mem_z, pad], dim=-1)
            mem_z = mem_z[:, :32]  # ensure correct size
            mem_z = mem_z.unsqueeze(0).expand(B, -1, -1)  # (B, k, mem_dim)
        else:
            mem_z = torch.zeros(B, self.memory_top_k, 32, device=device)

        # 3. build prev proposal features
        dA_norm = prev_delta_A.norm().item() / 768.0 if prev_delta_A is not None else 0.0
        dB_norm = prev_delta_B.norm().item() / 16.0 if prev_delta_B is not None else 0.0
        target_oh = torch.zeros(3, device=device)
        target_oh[min(prev_target, 2)] = 1.0
        # match episode format: [dA_norm, weight, target]
        prop_feat = torch.tensor(
            [[dA_norm, prev_weight, float(prev_target)]],
            device=device,
        ).expand(B, -1)

        # 4. correction policy forward
        return self.policy(
            error_state_tensor, mem_z, prop_feat, core_z, mode=mode,
        )


# ---- Rollback Controller (C1) ----

class RollbackController:
    """Experiment controller: saves pre-modification state and restores on failure.

    This is NOT a model component — it's the experiment controller that
    implements the C1 (rollback) ablation baseline.
    """

    def __init__(self):
        self.saved_state = None

    def save(self, network) -> None:
        """Snapshot adapter parameters before modification."""
        self.saved_state = network.snapshot_parameters()

    def rollback(self, network) -> bool:
        """Restore adapter parameters to pre-modification state."""
        if self.saved_state is not None:
            network.restore_parameters(self.saved_state)
            self.saved_state = None
            return True
        return False
