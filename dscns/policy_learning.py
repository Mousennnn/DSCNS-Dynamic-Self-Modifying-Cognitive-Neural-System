"""Policy Learning (v0.5.3 / Phase 5.5).

The core training mechanism that uses experience to update the modification
policy.  This is what was MISSING in v0.5.2 — memory was stored and retrieved
but never trained the policy.

    L_policy = L_outcome
             + λ1 × L_contrastive
             + λ2 × L_avoid
             + λ3 × L_reuse
             + λ4 × L_stability

Components:
  ModificationPolicyLearner    -- coordinates all losses
  ExperiencePolicyUpdater      -- applies updates from experience
  OutcomeWeightedPolicyLoss    -- weighted by outcome quality
  ContrastiveExperienceLoss    -- success > failure for similar contexts
  FailureAvoidanceLoss         -- lower P(repeat failed modification)
  SuccessReuseLoss             -- higher P(reuse successful modification)
  StabilityLoss                -- D_KL(π_new || π_old) < δ
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveExperienceLoss(nn.Module):
    """L_contrastive: for similar contexts, success score > failure score.

    For success experience E_s and failure experience E_f with similar
    context (Sim(C_s, C_f) ≈ 1):

        L = max(0, margin - Score(E_s) + Score(E_f))

    This teaches the policy to prefer successful modifications over
    failed ones in similar situations.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, score_success: torch.Tensor,
                score_failure: torch.Tensor,
                similarity: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            score_success: (B,) policy scores for success experiences
            score_failure: (B,) policy scores for failure experiences
            similarity: (B,) context similarity [0, 1]

        Returns:
            Scalar contrastive loss.
        """
        # only apply when contexts are similar enough
        weight = similarity.clamp(min=0.0, max=1.0)
        loss_per = F.relu(self.margin - (score_success - score_failure))
        weighted_loss = weight * loss_per
        return weighted_loss.mean() if weighted_loss.numel() > 0 else torch.tensor(0.0)


class FailureAvoidanceLoss(nn.Module):
    """L_avoid: lower P(repeat failed modification).

    For a failed experience E_f, the policy should output low probability
    for the same modification:

        L_avoid = -log(1 - P(repeat(E_f)))

    Implemented as: cross-entropy where the target is "don't repeat".
    """

    def forward(self, target_probs: torch.Tensor,
                failed_targets: torch.Tensor,
                weight: float = 1.0) -> torch.Tensor:
        """Compute failure avoidance loss.

        Args:
            target_probs: (B, K) current target probability distribution
            failed_targets: (B,) target groups that previously failed
            weight: loss weight

        Returns:
            Scalar avoidance loss.
        """
        if target_probs.size(0) == 0:
            return torch.tensor(0.0)

        # for each failed target, penalize high probability
        B, K = target_probs.shape
        # create soft target: low probability for failed targets
        target_mask = F.one_hot(failed_targets.long(), K).float()
        # we want 1 - target_mask to have high probability
        # equivalent to: -log(1 - p_failed)
        p_failed = (target_probs * target_mask).sum(dim=-1)
        p_failed = p_failed.clamp(1e-6, 1.0 - 1e-6)
        loss = -torch.log(1.0 - p_failed)
        return weight * loss.mean()


class SuccessReuseLoss(nn.Module):
    """L_reuse: higher P(reuse successful modification).

    For a successful experience E_s, the policy should output high
    probability for the same modification:

        L_reuse = -log(P(reuse(E_s)))
    """

    def forward(self, target_probs: torch.Tensor,
                success_targets: torch.Tensor,
                weight: float = 1.0) -> torch.Tensor:
        """Compute success reuse loss.

        Args:
            target_probs: (B, K) current target probability distribution
            success_targets: (B,) target groups that previously succeeded
            weight: loss weight

        Returns:
            Scalar reuse loss.
        """
        if target_probs.size(0) == 0:
            return torch.tensor(0.0)

        B, K = target_probs.shape
        target_mask = F.one_hot(success_targets.long(), K).float()
        p_success = (target_probs * target_mask).sum(dim=-1)
        p_success = p_success.clamp(1e-6, 1.0 - 1e-6)
        loss = -torch.log(p_success)
        return weight * loss.mean()


class StabilityLoss(nn.Module):
    """L_stability: D_KL(π_new || π_old) < δ.

    Prevents the policy from changing too rapidly.
    Implemented as forward KL divergence between new and old
    target distributions.
    """

    def __init__(self, max_kl: float = 0.5):
        super().__init__()
        self.max_kl = max_kl

    def forward(self, new_probs: torch.Tensor,
                old_probs: torch.Tensor) -> torch.Tensor:
        """Compute stability loss (KL divergence).

        Args:
            new_probs: (B, K) new target distribution
            old_probs: (B, K) old target distribution

        Returns:
            Scalar KL loss, clamped to max_kl.
        """
        # KL(new || old) = Σ new * log(new / old)
        new_p = new_probs.clamp(1e-8, 1.0)
        old_p = old_probs.clamp(1e-8, 1.0)
        kl = (new_p * (new_p / old_p).log()).sum(dim=-1)
        kl = kl.clamp(max=self.max_kl)
        return kl.mean()


class ModificationPolicyLearner:
    """Coordinates all policy losses and applies updates.

    Usage per round:
        1. Get current policy output (target_probs, magnitude, etc.)
        2. After outcome is known, compute losses
        3. Call step() to update policy

    Supports ablation modes:
        - full: all losses
        - no_contrastive: skip L_contrastive
        - no_avoid: skip L_avoid
        - no_reuse: skip L_reuse
        - no_stability: skip L_stability
    """

    def __init__(self, lr: float = 3e-4,
                 lambda_contrastive: float = 1.0,
                 lambda_avoid: float = 0.5,
                 lambda_reuse: float = 0.5,
                 lambda_stability: float = 0.1,
                 contrastive_margin: float = 0.1,
                 stability_max_kl: float = 0.5,
                 device: str = "cpu"):
        self.lr = lr
        self.lambda_contrastive = lambda_contrastive
        self.lambda_avoid = lambda_avoid
        self.lambda_reuse = lambda_reuse
        self.lambda_stability = lambda_stability
        self.device = device

        # loss functions
        self.contrastive_loss = ContrastiveExperienceLoss(margin=contrastive_margin)
        self.avoid_loss = FailureAvoidanceLoss()
        self.reuse_loss = SuccessReuseLoss()
        self.stability_loss = StabilityLoss(max_kl=stability_max_kl)

        # experience pairs for contrastive learning
        self.success_pairs: List[Dict[str, Any]] = []
        self.failure_pairs: List[Dict[str, Any]] = []

        # old policy distribution for stability
        self.old_target_probs: Optional[torch.Tensor] = None

        # loss tracking
        self.loss_history: List[Dict[str, float]] = []
        self.total_steps = 0

    def set_old_probs(self, probs: torch.Tensor) -> None:
        """Snapshot the current policy distribution before update."""
        self.old_target_probs = probs.detach().clone()

    def compute_loss(
        self,
        new_target_probs: torch.Tensor,      # (B, K)
        outcome: str,                          # success/failure/recovery
        target_group: int,                     # which target was used
        old_target_probs: Optional[torch.Tensor] = None,  # (B, K) for stability
        contrastive_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
        failed_targets: Optional[torch.Tensor] = None,  # (N,) targets that failed
        success_targets: Optional[torch.Tensor] = None,  # (N,) targets that succeeded
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined policy loss.

        Returns dict with 'total' and individual losses.
        """
        losses = {"total": torch.tensor(0.0, device=new_target_probs.device)}

        # L_contrastive: success vs failure for similar contexts
        if contrastive_pairs and len(contrastive_pairs) > 0:
            s_scores = torch.stack([p[0] for p in contrastive_pairs])
            f_scores = torch.stack([p[1] for p in contrastive_pairs])
            sims = torch.tensor([p[2] for p in contrastive_pairs],
                                device=new_target_probs.device)
            l_contrastive = self.contrastive_loss(s_scores, f_scores, sims)
            losses["contrastive"] = l_contrastive
            losses["total"] = losses["total"] + self.lambda_contrastive * l_contrastive

        # L_avoid: don't repeat failed targets
        if failed_targets is not None and failed_targets.numel() > 0:
            l_avoid = self.avoid_loss(new_target_probs, failed_targets)
            losses["avoid"] = l_avoid
            losses["total"] = losses["total"] + self.lambda_avoid * l_avoid

        # L_reuse: do reuse successful targets
        if success_targets is not None and success_targets.numel() > 0:
            l_reuse = self.reuse_loss(new_target_probs, success_targets)
            losses["reuse"] = l_reuse
            losses["total"] = losses["total"] + self.lambda_reuse * l_reuse

        # L_stability: don't change too fast
        old_p = old_target_probs if old_target_probs is not None else self.old_target_probs
        if old_p is not None:
            l_stability = self.stability_loss(new_target_probs, old_p.detach())
            losses["stability"] = l_stability
            losses["total"] = losses["total"] + self.lambda_stability * l_stability

        self.total_steps += 1
        self.loss_history.append({
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in losses.items()
        })
        return losses

    def update_experience_pairs(
        self,
        success_context: Optional[torch.Tensor] = None,
        success_score: Optional[torch.Tensor] = None,
        failure_context: Optional[torch.Tensor] = None,
        failure_score: Optional[torch.Tensor] = None,
        similarity: float = 0.0,
    ) -> None:
        """Store a contrastive pair for future training."""
        if (success_context is not None and success_score is not None and
                failure_context is not None and failure_score is not None):
            self.success_pairs.append({
                "context": success_context.detach(),
                "score": success_score.detach(),
            })
            self.failure_pairs.append({
                "context": failure_context.detach(),
                "score": failure_score.detach(),
                "similarity": similarity,
            })
            # keep bounded
            max_pairs = 200
            if len(self.success_pairs) > max_pairs:
                self.success_pairs = self.success_pairs[-max_pairs:]
                self.failure_pairs = self.failure_pairs[-max_pairs:]

    def loss_statistics(self) -> Dict[str, Any]:
        """Summary of loss history."""
        if not self.loss_history:
            return {"n_steps": 0}
        stats = {}
        for key in self.loss_history[0]:
            vals = [h.get(key, 0.0) for h in self.loss_history]
            arr = np.array(vals)
            stats[key] = {"mean": float(arr.mean()), "std": float(arr.std()),
                          "last": float(arr[-1])}
        stats["n_steps"] = self.total_steps
        return stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "lambdas": {
                "contrastive": self.lambda_contrastive,
                "avoid": self.lambda_avoid,
                "reuse": self.lambda_reuse,
                "stability": self.lambda_stability,
            },
            "loss_statistics": self.loss_statistics(),
            "n_pairs": len(self.success_pairs),
        }
