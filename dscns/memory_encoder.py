"""Memory encoder for P5.1/v0.5.1 — Memory-Conditioned Outcome Learning.

Encodes episodic modification records into compact memory representations
that enter the correction policy and future modification proposals.

Architecture:
  Episode → [proposal_encoder, outcome_encoder, context_encoder] → z_memory

The memory encoder produces a fixed-dim (default 32) representation that
captures the essential structure of a modification episode:
  - What was attempted (proposal)
  - What happened (outcome/error)
  - What context it happened in (context)

This is a model-side component, not an experiment controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---- Episode data structure (task specification §9) ----

@dataclass
class ModificationEpisode:
    """Complete modification episode record for memory storage.

    Stores the full transition: state + proposal → outcome, so that
    the system can learn state→modification→consequence mappings.
    """
    # identity
    round_id: int = 0
    seed: int = 0

    # context
    context_embedding: Optional[Any] = None   # (context_dim,) task context
    context_domain: str = ""

    # pre-state
    state_before: Optional[Any] = None        # (256,) core z at generation

    # proposal
    proposal: Optional[Any] = None            # delta_W_A, delta_W_B
    proposal_norm: float = 0.0
    target: int = 0
    weight: float = 0.0                       # w_t applied magnitude

    # effective modification
    effective_delta_norm: float = 0.0

    # post-state
    state_after: Optional[Any] = None

    # outcome
    score_before: float = 0.0
    score_after: float = 0.0
    delta_score: float = 0.0
    error_state: Optional[Any] = None         # ErrorState

    # classification
    outcome: str = "neutral"                  # success / failure / partial_success
    category: str = "success"                 # success / failure / recovery

    # correction (filled after failure)
    correction_applied: bool = False
    correction_norm: float = 0.0
    correction_strength: float = 0.0

    # recovery
    recovery_score: float = 0.0

    def to_feature_vector(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Convert episode to fixed-dim feature vector for similarity search.

        Returns (feat_dim,) tensor capturing:
          [proposal_norm(1), weight(1), target_onehot(3), delta_score(1),
           error_components(8), outcome_onehot(3), correction_norm(1)]
        = 15 features total
        """
        target_oh = torch.zeros(3, device=device)
        target_oh[min(self.target, 2)] = 1.0

        outcome_oh = torch.zeros(3, device=device)
        if self.category == "success":
            outcome_oh[0] = 1.0
        elif self.category == "failure":
            outcome_oh[1] = 1.0
        elif self.category == "recovery":
            outcome_oh[2] = 1.0

        err = torch.zeros(8, device=device)
        if self.error_state is not None:
            if hasattr(self.error_state, "to_tensor"):
                err = self.error_state.to_tensor().to(device)
            elif isinstance(self.error_state, torch.Tensor):
                err = self.error_state.to(device)[:8]

        feat = torch.cat([
            torch.tensor([self.proposal_norm, self.weight], device=device),
            target_oh,
            torch.tensor([self.delta_score], device=device),
            err,
            outcome_oh,
            torch.tensor([self.correction_norm], device=device),
        ])
        return feat


# ---- Memory Encoder Module ----

class MemoryEncoder(nn.Module):
    """Encode modification episodes into compact memory representations.

    Produces z_memory ∈ R^{memory_dim} from episode features + optional
    core_z from the plasticity module.

    Two encoding paths:
      1. Feature-based: episode features → MLP → z_feat
      2. Core-based: core_z (from plasticity) → linear → z_core
    Final: z_memory = normalize(z_feat + z_core)
    """

    def __init__(self, feature_dim: int = 15, memory_dim: int = 32,
                 core_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.memory_dim = memory_dim

        # feature-based encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        # core-based encoder (from plasticity module's z)
        self.core_encoder = nn.Sequential(
            nn.Linear(core_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        # fusion gate
        self.gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid(),
        )

    def forward(self, episode_features: torch.Tensor,
                core_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode episodes into memory representations.

        Args:
            episode_features: (B, feature_dim) episode feature vectors
            core_z: (B, core_dim) optional core representations from plasticity

        Returns:
            z_memory: (B, memory_dim) memory representations
        """
        z_feat = self.feature_encoder(episode_features)  # (B, memory_dim)

        if core_z is not None:
            z_core = self.core_encoder(core_z)           # (B, memory_dim)
            gate = self.gate(torch.cat([z_feat, z_core], dim=-1))  # (B, memory_dim)
            z_memory = gate * z_feat + (1 - gate) * z_core
        else:
            z_memory = z_feat

        # L2 normalize for stable similarity computation
        z_memory = F.normalize(z_memory, p=2, dim=-1)
        return z_memory

    def encode_single(self, episode: ModificationEpisode,
                      device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Encode a single episode. Returns (1, memory_dim)."""
        feat = episode.to_feature_vector(device).unsqueeze(0)  # (1, feat_dim)
        core_z = None
        if episode.state_before is not None:
            if isinstance(episode.state_before, torch.Tensor):
                core_z = episode.state_before.to(device).unsqueeze(0)
            else:
                core_z = torch.tensor(
                    episode.state_before, dtype=torch.float32, device=device
                ).unsqueeze(0)
        return self.forward(feat, core_z)


# ---- Multi-Similarity Retrieval ----

class MultiSimilarityRetriever:
    """Retrieve episodes using weighted multi-similarity (task spec §10).

    Similarity = λ_c * S_context + λ_p * S_proposal + λ_e * S_error + λ_t * S_target

    Where each S_* is cosine similarity on the respective embedding space.
    """

    def __init__(self, top_k: int = 8,
                 lambda_context: float = 0.3,
                 lambda_proposal: float = 0.3,
                 lambda_error: float = 0.2,
                 lambda_target: float = 0.2,
                 similarity_threshold: float = 0.5):
        self.top_k = top_k
        self.lambda_context = lambda_context
        self.lambda_proposal = lambda_proposal
        self.lambda_error = lambda_error
        self.lambda_target = lambda_target
        self.similarity_threshold = similarity_threshold

    def retrieve(self,
                 query_context: Optional[torch.Tensor],
                 query_proposal: Optional[torch.Tensor],
                 query_error: Optional[torch.Tensor],
                 query_target: Optional[int],
                 episodes: List[ModificationEpisode],
                 memory_z_cache: Optional[Dict[int, torch.Tensor]] = None,
                 ) -> List[Tuple[float, ModificationEpisode]]:
        """Retrieve top-k episodes by weighted multi-similarity.

        Returns list of (similarity, episode) sorted by similarity descending.
        """
        if not episodes:
            return []

        scores = []
        for ep in episodes:
            sim = self._compute_similarity(
                query_context, query_proposal, query_error, query_target,
                ep, memory_z_cache)
            scores.append((sim, ep))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:self.top_k]

    def _compute_similarity(self, q_ctx, q_prop, q_err, q_tgt,
                            ep: ModificationEpisode,
                            z_cache: Optional[Dict] = None) -> float:
        """Weighted multi-component similarity."""
        s_ctx = self._cosine_sim(q_ctx, self._episode_to_ctx(ep))
        s_prop = self._cosine_sim(q_prop, self._episode_to_prop(ep))
        s_err = self._cosine_sim(q_err, self._episode_to_err(ep))
        s_tgt = 1.0 if q_tgt is not None and q_tgt == getattr(ep, "target", getattr(ep, "target_group", -1)) else 0.0

        return (self.lambda_context * s_ctx +
                self.lambda_proposal * s_prop +
                self.lambda_error * s_err +
                self.lambda_target * s_tgt)

    def find_similar_failures(self, episodes: List[ModificationEpisode],
                              query_error: Optional[torch.Tensor] = None,
                              query_context: Optional[torch.Tensor] = None,
                              threshold: Optional[float] = None) -> List[ModificationEpisode]:
        """Find episodes with similar failures (for RFR-similar calculation).

        An episode is a 'similar failure' if:
          1. It's a failure outcome
          2. Multi-similarity with query > threshold
        """
        threshold = threshold or self.similarity_threshold
        failures = [ep for ep in episodes if ep.category == "failure"]
        if not failures:
            return []

        scored = []
        for ep in failures:
            sim = self._compute_similarity(
                query_context, None, query_error, None, ep, None)
            if sim >= threshold:
                scored.append((sim, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored]

    @staticmethod
    def _cosine_sim(a: Optional[torch.Tensor],
                    b: Optional[torch.Tensor]) -> float:
        """Cosine similarity between two vectors, handling None and mismatched dims."""
        if a is None or b is None:
            return 0.0
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).float()
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b).float()
        a = a.flatten().float()
        b = b.flatten().float()
        # ensure same device
        target_dev = a.device
        b = b.to(target_dev)
        # pad shorter to match dimension
        if a.size(0) != b.size(0):
            max_len = max(a.size(0), b.size(0))
            if a.size(0) < max_len:
                a = torch.cat([a, torch.zeros(max_len - a.size(0), device=a.device)])
            if b.size(0) < max_len:
                b = torch.cat([b, torch.zeros(max_len - b.size(0), device=b.device)])
        norm = a.norm() * b.norm()
        if norm < 1e-12:
            return 0.0
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    @staticmethod
    def _episode_to_ctx(ep) -> Optional[torch.Tensor]:
        """Extract context from episode (handles both ModificationEpisode and EpisodicModificationRecord)."""
        if hasattr(ep, "context_embedding") and ep.context_embedding is not None:
            return ep.context_embedding
        if hasattr(ep, "core_z") and ep.core_z is not None:
            return ep.core_z
        if hasattr(ep, "state_pooled") and ep.state_pooled is not None:
            return ep.state_pooled
        return None

    @staticmethod
    def _episode_to_prop(ep) -> Optional[torch.Tensor]:
        """Extract proposal features from episode (handles both types)."""
        if hasattr(ep, "proposal") and ep.proposal is not None:
            if isinstance(ep.proposal, dict):
                dA = ep.proposal.get("delta_W_A")
                dB = ep.proposal.get("delta_W_B")
                if dA is not None and dB is not None:
                    if isinstance(dA, torch.Tensor) and isinstance(dB, torch.Tensor):
                        return torch.cat([dA.flatten(), dB.flatten()]).float()
        # fallback: use available numeric features
        p_norm = getattr(ep, "proposal_norm", getattr(ep, "delta_norm", 0.0))
        w = getattr(ep, "weight", getattr(ep, "magnitude", 0.0))
        t = getattr(ep, "target", getattr(ep, "target_group", 0))
        return torch.tensor([p_norm, w, float(t)], dtype=torch.float32)

    @staticmethod
    def _episode_to_err(ep) -> Optional[torch.Tensor]:
        """Extract error state from episode (handles both types)."""
        err = getattr(ep, "error_state", None)
        if err is not None and hasattr(err, "to_tensor"):
            return err.to_tensor()
        return None


# ---- Memory-to-Policy Encoder ----

class MemoryPolicyEncoder(nn.Module):
    """Encode retrieved memory summaries into policy-conditioning signals.

    Takes top-k memory representations and produces:
      - z_memory_summary: (memory_dim,) summary for correction policy
      - memory_attention_weights: (k,) attention over retrieved episodes
    """
    def __init__(self, memory_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.projector = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(self, memory_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode retrieved memory representations.

        Args:
            memory_z: (k, memory_dim) or (B, k, memory_dim) retrieved z_memory

        Returns:
            z_summary: (B, memory_dim) or (memory_dim,) weighted summary
            attn_weights: (B, k) or (k,) attention weights
        """
        if memory_z.dim() == 2:
            # single sample: (k, memory_dim)
            attn_logits = self.attention(memory_z).squeeze(-1)  # (k,)
            attn_weights = F.softmax(attn_logits, dim=0)        # (k,)
            z_summary = torch.matmul(attn_weights, memory_z)    # (memory_dim,)
            return self.projector(z_summary), attn_weights
        elif memory_z.dim() == 3:
            # batched: (B, k, memory_dim)
            attn_logits = self.attention(memory_z).squeeze(-1)  # (B, k)
            attn_weights = F.softmax(attn_logits, dim=1)        # (B, k)
            z_summary = torch.bmm(
                attn_weights.unsqueeze(1), memory_z).squeeze(1)  # (B, memory_dim)
            return self.projector(z_summary), attn_weights
        else:
            raise ValueError(f"Expected 2D or 3D memory_z, got {memory_z.dim()}D")
