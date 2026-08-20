"""Intrinsic plasticity module (Phase 5 / 5.1, report sections 4-6).

Level-3 intrinsic self-modification: the model's internal state directly
produces a change to its own parameters:

    delta_theta_t = P_phi(h_t, stats(theta_t), s_t)
    theta_{t+1}   = theta_t + alpha * delta_theta_t

P5.1 extension (report sections 5-9): the module additionally outputs
  - a self-determined *magnitude* m_t ∈ [m_min, m_max]
  - a *target group* selection over shape-compatible adapter projections
  - accepts *error state* and *retrieved memory* as optional inputs
and returns a ModificationProposal instead of just a delta dict.

The original P5 `forward()` API is preserved for backward compatibility.
P5.1 uses the new `generate_proposal()` method.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

from .error_correction import ErrorEncoder, ErrorState


# Target groups (shape-compatible adapter projections, per P5.1 report §9):
#   0 = attn c_proj lora_A (all 12 layers, shape 16×768)
#   1 = attn c_proj lora_B (all 12 layers, shape 768×16)
#   2 = mlp  c_proj lora_B (all 12 layers, shape 768×16)
NUM_TARGET_GROUPS = 3
TARGET_NAMES = ["attn_lora_A", "attn_lora_B", "mlp_lora_B"]


class IntrinsicPlasticityModule(nn.Module):
    """Generate delta_theta from hidden states, parameter statistics and meta.

    Args:
        hidden_dim: base model hidden size (768 for GPT-2 small).
        adapter_dim: LoRA rank (16 in this prototype).
        meta_dim: self-state meta vector dimension (s_t).
        plasticity_rank: rank of the low-rank delta decomposition.
        use_hidden / use_param_stats / use_meta: state-component ablation.
        modulation_strength_init: initial value of the learnable strength.
    """

    def __init__(self, hidden_dim: int = 768, adapter_dim: int = 16,
                 meta_dim: int = 32, plasticity_rank: int = 8,
                 use_hidden: bool = True, use_param_stats: bool = True,
                 use_meta: bool = True, modulation_strength_init: float = 0.05,
                 # ---- P5.1 extension ----
                 p51: bool = False,
                 m_min: float = 0.02, m_max: float = 1.0,
                 m_init_bias: float = -3.0,
                 error_dim: int = 32,
                 num_target_groups: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.meta_dim = meta_dim
        self.plasticity_rank = plasticity_rank
        self.use_hidden = use_hidden
        self.use_param_stats = use_param_stats
        self.use_meta = use_meta

        # ---- state extractor: h -> z_h ----
        self.state_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )

        # ---- parameter-statistics encoder: stats(theta) -> z_theta ----
        # input: [mean, std, min, max] = 4 dims (low-dimensional approx)
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # ---- meta encoder: s -> z_s ----
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # ---- fusion: [z_h, z_theta, z_s] -> z ----
        fusion_input_dim = (128 if use_hidden else 0) + \
                           (64 if use_param_stats else 0) + \
                           (64 if use_meta else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ---- low-rank delta generators (delta W = U V^T) ----
        # delta_W_A: (hidden_dim, adapter_dim), delta_W_B: (adapter_dim, hidden_dim)
        self.delta_W_A_U_generator = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, hidden_dim * plasticity_rank), nn.Tanh(),
        )
        self.delta_W_A_V_generator = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, adapter_dim * plasticity_rank), nn.Tanh(),
        )
        self.delta_W_B_U_generator = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, adapter_dim * plasticity_rank), nn.Tanh(),
        )
        self.delta_W_B_V_generator = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, hidden_dim * plasticity_rank), nn.Tanh(),
        )

        # learnable global modulation strength (small, conservative init)
        self.modulation_strength = nn.Parameter(
            torch.tensor(float(modulation_strength_init)))

        # statistics
        self.num_modifications = 0
        self.total_delta_norm = 0.0
        self._num_forward_calls = 0

        # ---- P5.1 extension (report sections 5-9) ----
        self.p51 = p51
        self.m_min = m_min
        self.m_max = m_max
        if p51:
            # Heads take core z (256) + error embedding + memory embedding
            head_input_dim = 256 + error_dim + error_dim
            # init so sigmoid produces m_init_value=0.05 (between m_min and m_max)
            m_init_value = 0.05
            init_frac = (m_init_value - m_min) / (m_max - m_min + 1e-8)
            init_frac = max(min(init_frac, 0.9999), 0.0001)  # clamp to valid range
            init_logit = torch.tensor(
                float(np.log(init_frac / (1.0 - init_frac + 1e-12)))
            ) if init_frac > 0 else torch.tensor(-3.0)
            self.magnitude_head = nn.Sequential(
                nn.Linear(head_input_dim, 16), nn.Tanh(),
                nn.Linear(16, 1),
            )
            with torch.no_grad():
                self.magnitude_head[-1].bias.fill_(float(init_logit))
            self.target_head = nn.Sequential(
                nn.Linear(head_input_dim, 32), nn.ReLU(),
                nn.Linear(32, num_target_groups),
            )
            self.error_encoder = ErrorEncoder(error_dim=error_dim)
            self.error_dim = error_dim
            # NOTE: fusion stays at 256 input (P5 backward-compatible)

    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor,
                current_params: Dict[str, torch.Tensor],
                meta_info: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Produce parameter changes from internal state.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            current_params: {'W_A': tensor, 'W_B': tensor} (flattened ok)
            meta_info: (batch, meta_dim)
            mask: (batch, seq_len)

        Returns:
            dict with 'delta_W_A' (hidden,adapter), 'delta_W_B' (adapter,hidden),
            'modulation_strength', 'per_sample' deltas and 'components'.
        """
        batch_size = hidden_states.size(0)

        # 1. pooled hidden state h -> z_h
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled_h = (hidden_states * mask_exp).sum(1) / \
                mask_exp.sum(1).clamp(min=1e-9)
        else:
            pooled_h = hidden_states.mean(dim=1)  # (B, H)
        if self.use_hidden:
            z_h = self.state_extractor(pooled_h)  # (B, 128)
        else:
            z_h = torch.zeros(batch_size, 128, device=hidden_states.device)

        # 2. parameter statistics stats(theta) -> z_theta
        if self.use_param_stats:
            param_stats = self._encode_param_stats(current_params)  # (4,)
            param_stats = param_stats.unsqueeze(0).expand(batch_size, -1)
            z_theta = self.param_encoder(param_stats)  # (B, 64)
        else:
            z_theta = torch.zeros(batch_size, 64, device=hidden_states.device)

        # 3. meta info s -> z_s
        meta_info = meta_info.to(hidden_states.device)
        if self.use_meta:
            z_s = self.meta_encoder(meta_info)  # (B, 64)
        else:
            z_s = torch.zeros(batch_size, 64, device=hidden_states.device)

        # 4. fusion
        z = self.fusion(torch.cat([z_h, z_theta, z_s], dim=-1))  # (B, 256)

        # 5. low-rank delta generation  (delta W = U V^T)
        U_A = self.delta_W_A_U_generator(z).view(
            batch_size, self.hidden_dim, self.plasticity_rank)
        V_A = self.delta_W_A_V_generator(z).view(
            batch_size, self.adapter_dim, self.plasticity_rank)
        delta_W_A_full = torch.bmm(U_A, V_A.transpose(1, 2))  # (B, H, r)

        U_B = self.delta_W_B_U_generator(z).view(
            batch_size, self.adapter_dim, self.plasticity_rank)
        V_B = self.delta_W_B_V_generator(z).view(
            batch_size, self.hidden_dim, self.plasticity_rank)
        delta_W_B_full = torch.bmm(U_B, V_B.transpose(1, 2))  # (B, r, H)

        # 6. global modulation strength
        strength = self.modulation_strength.abs()

        # 7. per-sample (for analyses) and consensus (applied) deltas
        per_sample = {
            "delta_W_A": delta_W_A_full * strength,   # (B, H, r)
            "delta_W_B": delta_W_B_full * strength,   # (B, r, H)
        }
        delta_W_A = delta_W_A_full.mean(dim=0) * strength   # (H, r)
        delta_W_B = delta_W_B_full.mean(dim=0) * strength   # (r, H)

        # 8. statistics
        self.num_modifications += 1
        self.total_delta_norm += (delta_W_A.norm().item() +
                                  delta_W_B.norm().item())
        self._num_forward_calls += 1

        return {
            "delta_W_A": delta_W_A,
            "delta_W_B": delta_W_B,
            "modulation_strength": strength.item(),
            "per_sample": per_sample,
            "components": {
                "z_h": z_h.detach(),
                "z_theta": z_theta.detach(),
                "z_s": z_s.detach(),
                "z": z.detach(),
                "pooled_h": pooled_h.detach(),
            },
        }

    # ------------------------------------------------------------------ #
    def forward_from_pooled(self, pooled_hidden: torch.Tensor,
                            current_params: Dict[str, torch.Tensor],
                            meta_info: torch.Tensor) -> Dict[str, Any]:
        """Replay path used by the P5-C trainer (states already pooled).

        pooled_hidden: (B, hidden_dim).  Equivalent to forward() on a
        (B, 1, hidden_dim) tensor, so the trainer can replay recorded cases
        without storing full hidden-state sequences.
        """
        return self.forward(
            pooled_hidden.unsqueeze(1), current_params, meta_info, mask=None)

    # ------------------------------------------------------------------ #
    def _encode_param_stats(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Low-dimensional stats(theta) = [mean, std, min, max]."""
        W_A = params["W_A"].detach().flatten().float()
        W_B = params["W_B"].detach().flatten().float()
        all_params = torch.cat([W_A, W_B])
        stats = torch.tensor([
            all_params.mean().item(),
            all_params.std().item(),
            all_params.min().item(),
            all_params.max().item(),
        ], device=all_params.device)
        return stats

    def reset_statistics(self) -> None:
        self.num_modifications = 0
        self.total_delta_norm = 0.0
        self._num_forward_calls = 0

    def get_statistics(self) -> Dict[str, float]:
        return {
            "num_modifications": self.num_modifications,
            "num_forward_calls": self._num_forward_calls,
            "avg_delta_norm": self.total_delta_norm / max(1, self.num_modifications),
            "modulation_strength": self.modulation_strength.item(),
            "p51": self.p51,
        }

    # ------------------------------------------------------------------ #
    # P5.1: generate_proposal  (report sections 5-9)
    # ------------------------------------------------------------------ #
    def generate_proposal(
        self,
        hidden_states: torch.Tensor,
        current_params: Dict[str, torch.Tensor],
        meta_info: torch.Tensor,
        error_state: Optional[ErrorState] = None,
        memory_z: Optional[torch.Tensor] = None,   # (B, error_dim) from memory
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """P5.1 full proposal: delta + magnitude + target + confidence.

        Returns a dict with keys:
          delta_W_A, delta_W_B (consensus), magnitude, target_group,
          confidence, modulation_strength, core_z (for memory storage).
        """
        assert self.p51, "generate_proposal requires p51=True"
        batch_size = hidden_states.size(0)

        # ---- shared core features ----
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled_h = (hidden_states * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        else:
            pooled_h = hidden_states.mean(dim=1)
        z_h = self.state_extractor(pooled_h) if self.use_hidden else \
            torch.zeros(batch_size, 128, device=hidden_states.device)

        param_stats = self._encode_param_stats(current_params)
        param_stats = param_stats.unsqueeze(0).expand(batch_size, -1)
        z_theta = self.param_encoder(param_stats) if self.use_param_stats else \
            torch.zeros(batch_size, 64, device=hidden_states.device)

        meta_info = meta_info.to(hidden_states.device)
        if meta_info.dim() == 1:
            meta_info = meta_info.unsqueeze(0).expand(batch_size, -1)
        z_s = self.meta_encoder(meta_info) if self.use_meta else \
            torch.zeros(batch_size, 64, device=hidden_states.device)

        # error embedding
        if error_state is not None:
            err_t = error_state.to_tensor().to(hidden_states.device)
            if err_t.dim() == 1:
                err_t = err_t.unsqueeze(0).expand(batch_size, -1)
            z_e = self.error_encoder(err_t)
        else:
            z_e = torch.zeros(batch_size, self.error_dim,
                               device=hidden_states.device)

        # memory summary embedding
        if memory_z is not None:
            z_mem = memory_z.to(hidden_states.device)
            if z_mem.dim() == 1:
                z_mem = z_mem.unsqueeze(0).expand(batch_size, -1)
            z_mem = z_mem[:, :self.error_dim]
        else:
            z_mem = torch.zeros(batch_size, self.error_dim,
                                device=hidden_states.device)

        # fusion (P5-compatible: core z from h+theta+meta)
        z = self.fusion(torch.cat([z_h, z_theta, z_s], dim=-1))  # (B, 256)

        # delta generation (uses core z, same as P5)
        U_A = self.delta_W_A_U_generator(z).view(
            batch_size, self.hidden_dim, self.plasticity_rank)
        V_A = self.delta_W_A_V_generator(z).view(
            batch_size, self.adapter_dim, self.plasticity_rank)
        delta_W_A = torch.bmm(U_A, V_A.transpose(1, 2)).mean(dim=0) * \
            self.modulation_strength.abs()

        U_B = self.delta_W_B_U_generator(z).view(
            batch_size, self.adapter_dim, self.plasticity_rank)
        V_B = self.delta_W_B_V_generator(z).view(
            batch_size, self.hidden_dim, self.plasticity_rank)
        delta_W_B = torch.bmm(U_B, V_B.transpose(1, 2)).mean(dim=0) * \
            self.modulation_strength.abs()

        # P5.1 heads: magnitude + target take extended input [z; error; memory]
        extended_z = torch.cat([z, z_e, z_mem], dim=-1)  # (B, 256+2*error_dim)

        # magnitude: m_min + sigmoid(z_m) * (m_max - m_min)
        m_logits = self.magnitude_head(extended_z)                    # (B, 1)
        m_raw = torch.sigmoid(m_logits)                               # (B, 1) ∈ (0,1)
        magnitude = self.m_min + m_raw * (self.m_max - self.m_min)   # ∈ [m_min, m_max]
        mag_scalar = float(magnitude.mean())

        # target group: softmax distribution, greedy selection at inference
        target_logits = self.target_head(extended_z)                  # (B, num_groups)
        target_probs = torch.softmax(target_logits, dim=-1)           # (B, num_groups)
        target_group = int(target_probs.mean(dim=0).argmax())         # most probable group
        confidence = float(target_probs.mean(dim=0).max())

        self.num_modifications += 1
        self.total_delta_norm += float(delta_W_A.norm()) + float(delta_W_B.norm())
        self._num_forward_calls += 1

        return {
            "delta_W_A": delta_W_A,              # consensus (H, r)
            "delta_W_B": delta_W_B,              # consensus (r, H)
            "magnitude": mag_scalar,              # m_t ∈ [m_min, m_max]
            "target_group": target_group,         # 0,1,2
            "target_probs": target_probs.mean(dim=0).detach(),  # (num_groups,)
            "confidence": confidence,
            "modulation_strength": self.modulation_strength.abs().item(),
            "core_z": z.detach().mean(dim=0),     # (256,) for memory storage
        }
