"""Correction generator (P5.2, design report sections 9-10, 21).

When a modification fails, the correction generator produces a corrective
signal that augments the NEXT round's modification proposal.

The correction is NOT simply -Δθ.  It blends a reversal component with a
learned new direction, conditioned on the error signal and memory context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class CorrectionSignal:
    """Output of CorrectionGenerator."""
    correction_W_A: Any = None        # (768, 16)
    correction_W_B: Any = None        # (16, 768)
    correction_weight: float = 0.0
    reversal_alpha: float = 0.0
    confidence: float = 0.0


class CorrectionGenerator(nn.Module):
    """Generate correction signals from error state and memory."""

    def __init__(self, input_dim: int = 128, correction_dim: int = 128):
        super().__init__()
        self.correction_dim = correction_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, correction_dim), nn.ReLU())
        self.corr_A_proj = nn.Linear(correction_dim, 768 * 16)
        self.corr_B_proj = nn.Linear(correction_dim, 16 * 768)
        self.alpha_head = nn.Sequential(nn.Linear(correction_dim, 8), nn.Sigmoid(), nn.Linear(8, 1))
        self.strength_head = nn.Sequential(nn.Linear(correction_dim, 8), nn.Sigmoid(), nn.Linear(8, 1))

    def forward(self, error_tensor, prev_delta_A, prev_delta_B,
                prev_weight, prev_target, memory_z):
        B = error_tensor.size(0)
        device = error_tensor.device

        prev_dA_norm = prev_delta_A.norm().unsqueeze(0).expand(B, 1) / 768.0
        prev_dB_norm = prev_delta_B.norm().unsqueeze(0).expand(B, 1) / 16.0
        tgt = torch.zeros(B, 3, device=device)
        tgt[:, min(prev_target, 2)] = 1.0

        enc_input = torch.cat([
            error_tensor, prev_dA_norm, prev_dB_norm,
            prev_weight, tgt, memory_z[:, :16],
        ], dim=-1)
        z = self.encoder(enc_input)

        corr_A = self.corr_A_proj(z).view(B, 768, 16) * 0.01
        corr_B = self.corr_B_proj(z).view(B, 16, 768) * 0.01
        alpha_rev = self.alpha_head(z).squeeze(-1)  # (B,)
        strength = self.strength_head(z).squeeze(-1)  # (B,)

        neg_A = -prev_delta_A.unsqueeze(0)
        neg_B = -prev_delta_B.unsqueeze(0)
        alpha_vec = alpha_rev.view(B, 1, 1)

        correction_A = ((neg_A * alpha_vec) + corr_A * (1 - alpha_vec)).mean(dim=0)
        correction_B = ((neg_B * alpha_vec) + corr_B * (1 - alpha_vec)).mean(dim=0)

        return {
            "correction_W_A": correction_A,
            "correction_W_B": correction_B,
            "correction_weight": float(strength.mean()),
            "reversal_alpha": float(alpha_rev.mean()),
        }
