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


# ====================================================================== #
# v0.5.2 additions — direction, outcome, trainable encoding, attention   #
# ====================================================================== #
# The v0.5.1 encoder was effectively frozen in practice: z_memory was a
# fixed function of episode features, so different memory conditions could
# collapse to the same representation.  The v0.5.2 fix makes the encoder
# TRAINABLE and richer:
#   * DirectionEncoder  — encodes the delta_theta *direction* (not just
#     its norm) via a PCA-like projection, enabling cosine similarity on
#     modification directions.
#   * OutcomeEmbedding  — trainable 8-dim embedding of the 3 outcome
#     classes (SUCCESS / FAILURE / RECOVERY), making memory attention
#     outcome-aware.
#   * TrainedMemoryEncoder — wraps MemoryEncoder and concatenates the
#     base z_memory with direction + outcome embeddings through a
#     trainable projection, so distinct memory conditions yield distinct
#     z_memory.
#   * OutcomeAwareAttention — replaces mean pooling over retrieved
#     episodes with scaled-dot-product attention whose keys include
#     context + error + direction + outcome embedding.
# ====================================================================== #


class DirectionEncoder(nn.Module):
    """Encode delta_theta *direction* via a PCA-like projection (v0.5.2).

    Input:  delta_W_A (..., 768, 16) and delta_W_B (..., 16, 768)
            (the two low-rank LoRA deltas of a modification).
    Output: z_direction (B, direction_dim=16), L2-normalized.

    Two projection paths (both trainable / learnable):
      1. *PCA path* — ``fit_pca(samples)`` computes the top-16 principal
         directions of historical modification directions from the data
         (SVD), and forward() projects onto that basis.
      2. *Learned path* — a compact MLP-style linear projection
         (per-matrix compression then combine), used by default.

    The result supports cosine similarity on modification direction via
    ``DirectionEncoder.cosine_similarity``.
    """

    def __init__(self, hidden_dim: int = 768, adapter_dim: int = 16,
                 direction_dim: int = 16, compress_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.direction_dim = direction_dim

        self.flat_A_dim = hidden_dim * adapter_dim      # 768*16 = 12288
        self.flat_B_dim = adapter_dim * hidden_dim      # 16*768 = 12288
        self.flat_dim = self.flat_A_dim + self.flat_B_dim

        # learned (PCA-like) projection — trainable
        self.proj_A = nn.Linear(self.flat_A_dim, compress_dim)
        self.proj_B = nn.Linear(self.flat_B_dim, compress_dim)
        self.combine = nn.Linear(compress_dim * 2, direction_dim)

        # PCA basis fitted from data: (direction_dim, flat_dim)
        self.pca_basis: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def _flatten(self, delta_W_A, delta_W_B) -> torch.Tensor:
        """Return (B, flat_dim) tensor from the two deltas."""
        A = delta_W_A
        B = delta_W_B
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).float()
        A = A.reshape(-1, self.flat_A_dim)
        B = B.reshape(-1, self.flat_B_dim)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"delta_W_A batch {A.shape[0]} != delta_W_B batch {B.shape[0]}")
        return torch.cat([A, B], dim=-1).float()

    # ------------------------------------------------------------------ #
    def forward(self, delta_W_A: torch.Tensor,
                delta_W_B: torch.Tensor) -> torch.Tensor:
        """Encode direction -> (B, direction_dim), L2-normalized."""
        flat = self._flatten(delta_W_A, delta_W_B)
        if self.pca_basis is not None:
            z = torch.matmul(flat, self.pca_basis.t())       # (B, direction_dim)
        else:
            z_a = torch.relu(self.proj_A(flat[:, :self.flat_A_dim]))
            z_b = torch.relu(self.proj_B(flat[:, self.flat_A_dim:]))
            z = self.combine(torch.cat([z_a, z_b], dim=-1))  # (B, direction_dim)
        return F.normalize(z, p=2, dim=-1)

    # ------------------------------------------------------------------ #
    def fit_pca(self, samples: List[Tuple[Any, Any]]) -> "DirectionEncoder":
        """Fit the PCA projection basis from historical modifications.

        Args:
            samples: list of (delta_W_A, delta_W_B) pairs (each (768,16)
                / (16,768) tensor, array, or flattened vector).

        The basis is the top-``direction_dim`` right singular vectors of
        the stacked (N, flat_dim) direction matrix, computed with economy
        SVD to keep memory bounded.
        """
        if not samples:
            return self
        import numpy as np

        flat_list = []
        for dA, dB in samples:
            f = self._flatten(dA, dB).detach().cpu().numpy()  # (1, D)
            flat_list.append(f)
        X = np.concatenate(flat_list, axis=0).astype(np.float32)  # (N, D)
        X = X - X.mean(axis=0, keepdims=True)

        k = min(self.direction_dim, X.shape[0], X.shape[1])
        if k == 0:
            return self
        # economy SVD on X^T: (D, N) -> U (D, k), keeps memory bounded
        U, s, _ = np.linalg.svd(X.T, full_matrices=False)
        basis = U[:, :k].T.astype(np.float32)  # (k, D)
        self.pca_basis = torch.from_numpy(basis)
        return self

    def clear_pca(self) -> None:
        self.pca_basis = None

    # ------------------------------------------------------------------ #
    def encode_single(self, delta_W_A, delta_W_B) -> torch.Tensor:
        """Encode one modification -> (direction_dim,)."""
        return self.forward(delta_W_A, delta_W_B).squeeze(0)

    @staticmethod
    def cosine_similarity(z_a: torch.Tensor,
                          z_b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two direction encodings.

        Accepts (d,), (B, d) or (1, d); returns matching-shaped tensor.
        """
        if z_a.dim() == 1:
            z_a = z_a.unsqueeze(0)
        if z_b.dim() == 1:
            z_b = z_b.unsqueeze(0)
        return F.cosine_similarity(z_a, z_b, dim=-1)


class OutcomeEmbedding(nn.Module):
    """Trainable 3-class outcome embedding (v0.5.2).

    Maps SUCCESS / FAILURE / RECOVERY to an 8-dim vector so memory
    attention and retrieval can be *outcome-aware*.  The embedding table
    is a trainable nn.Embedding (different outcomes -> different vectors,
    learned from data), unlike a hard-coded one-hot.
    """

    SUCCESS = 0
    FAILURE = 1
    RECOVERY = 2

    LABELS = {"success": SUCCESS, "failure": FAILURE, "recovery": RECOVERY}

    def __init__(self, num_classes: int = 3, embed_dim: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        # conservative init so classes start distinguishable but mild
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.1)

    # ------------------------------------------------------------------ #
    @classmethod
    def index_of(cls, outcome: Any) -> int:
        """Map a label string / int to the class index (unknown -> 0)."""
        if isinstance(outcome, str):
            return cls.LABELS.get(outcome.lower(), cls.SUCCESS)
        try:
            return int(outcome) % cls.num_classes
        except (TypeError, ValueError):
            return cls.SUCCESS

    def forward(self, outcome: Any) -> torch.Tensor:
        """Embed outcome labels -> (B, embed_dim).

        Accepts a single string ("success"), an int index, a list of
        labels/indices, or a tensor of indices.
        """
        if isinstance(outcome, str):
            idx = torch.tensor([self.index_of(outcome)], dtype=torch.long)
        elif isinstance(outcome, int):
            idx = torch.tensor([outcome], dtype=torch.long)
        elif isinstance(outcome, torch.Tensor):
            if outcome.dim() == 0:
                idx = outcome.long().unsqueeze(0)
            else:
                idx = outcome.long()
        elif isinstance(outcome, (list, tuple)):
            idx = torch.tensor([self.index_of(o) for o in outcome],
                               dtype=torch.long)
        else:
            idx = torch.tensor([self.SUCCESS], dtype=torch.long)
        emb = self.embedding(idx)                      # (B, embed_dim)
        return emb

    def embed(self, outcome: Any) -> torch.Tensor:
        """Single-label embed -> (embed_dim,)."""
        return self.forward(outcome).squeeze(0)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """(3, embed_dim) raw embedding table."""
        return self.embedding.weight.detach()


class TrainedMemoryEncoder(nn.Module):
    """Trainable memory encoder with direction + outcome (v0.5.2).

    Wraps the existing :class:`MemoryEncoder` (kept intact) and enriches
    its z_memory with:

      * z_direction — DirectionEncoder of the modification direction,
      * z_outcome   — OutcomeEmbedding of the episode outcome,

    then projects the concatenation through a trainable projection.

    KEY FIX vs v0.5.1: every component is a trainable nn.Parameter / layer,
    so different memory conditions (different directions / outcomes /
    episodes) produce measurably different z_memory, and gradients flow
    back into the encoder when the downstream policy learns.
    """

    def __init__(self, base_encoder: Optional[MemoryEncoder] = None,
                 feature_dim: int = 18, memory_dim: int = 32,
                 core_dim: int = 256, hidden_dim: int = 64,
                 hidden_dim_delta: int = 768, adapter_dim: int = 16,
                 direction_dim: int = 16, outcome_dim: int = 8,
                 enrich_dim: Optional[int] = None):
        super().__init__()
        self.memory_dim = memory_dim
        self.direction_dim = direction_dim
        self.outcome_dim = outcome_dim
        self.enrich_dim = enrich_dim if enrich_dim is not None else memory_dim

        # existing encoder, kept intact (created fresh if not provided).
        # NOTE: ModificationEpisode.to_feature_vector emits 18 features
        # (2+3+1+8+3+1), so the wrapper defaults to feature_dim=18 to be
        # consistent with encode_single.
        self.base_encoder = base_encoder if base_encoder is not None else \
            MemoryEncoder(feature_dim=feature_dim, memory_dim=memory_dim,
                          core_dim=core_dim, hidden_dim=hidden_dim)

        # trainable v0.5.2 components
        self.direction_encoder = DirectionEncoder(
            hidden_dim=hidden_dim_delta, adapter_dim=adapter_dim,
            direction_dim=direction_dim)
        self.outcome_embedding = OutcomeEmbedding(
            num_classes=3, embed_dim=outcome_dim)

        in_dim = memory_dim + direction_dim + outcome_dim
        self.enrich_projection = nn.Sequential(
            nn.Linear(in_dim, max(hidden_dim, self.enrich_dim)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim, self.enrich_dim), self.enrich_dim),
        )

    # ------------------------------------------------------------------ #
    def forward(self, episode_features: torch.Tensor,
                core_z: Optional[torch.Tensor] = None,
                delta_W_A: Optional[torch.Tensor] = None,
                delta_W_B: Optional[torch.Tensor] = None,
                outcome: Any = None) -> Dict[str, torch.Tensor]:
        """Enrich base z_memory with direction + outcome embeddings.

        Returns a dict:
          z_memory    (B, enrich_dim)  final enriched memory (normalized)
          z_base      (B, memory_dim)  original MemoryEncoder output
          z_direction (B, direction_dim)
          z_outcome   (B, outcome_dim)
        """
        B = episode_features.shape[0]
        device = episode_features.device

        z_base = self.base_encoder(episode_features, core_z)  # (B, memory_dim)

        if delta_W_A is not None and delta_W_B is not None:
            z_direction = self.direction_encoder(delta_W_A, delta_W_B)
        else:
            z_direction = torch.zeros(B, self.direction_dim, device=device)

        if outcome is not None:
            z_outcome = self.outcome_embedding(outcome).to(device)
            if z_outcome.shape[0] != B:
                # broadcast a single outcome over the batch
                z_outcome = z_outcome.expand(B, -1)
        else:
            z_outcome = torch.zeros(B, self.outcome_dim, device=device)

        z_cat = torch.cat([z_base, z_direction, z_outcome], dim=-1)
        z_memory = F.normalize(self.enrich_projection(z_cat), p=2, dim=-1)

        return {
            "z_memory": z_memory,
            "z_base": z_base,
            "z_direction": z_direction,
            "z_outcome": z_outcome,
        }

    # ------------------------------------------------------------------ #
    def encode_single(self, episode: ModificationEpisode,
                      delta_W_A=None, delta_W_B=None, outcome: Any = None,
                      device: torch.device = torch.device("cpu"),
                      ) -> Dict[str, torch.Tensor]:
        """Encode a single ModificationEpisode -> enriched z_memory dict."""
        feat = episode.to_feature_vector(device).unsqueeze(0)  # (1, feat_dim)
        core_z = None
        if episode.state_before is not None:
            if isinstance(episode.state_before, torch.Tensor):
                core_z = episode.state_before.to(device).unsqueeze(0)
            else:
                core_z = torch.tensor(
                    episode.state_before, dtype=torch.float32, device=device
                ).unsqueeze(0)
        if outcome is None:
            outcome = episode.category
        return self.forward(feat, core_z, delta_W_A, delta_W_B, outcome)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OutcomeAwareAttention(nn.Module):
    """Outcome-aware attention over retrieved episodes (v0.5.2).

    Replaces mean pooling of retrieved memories with scaled dot-product
    attention:

        q = W_q(context_embedding)                # current context
        k_i = W_k([context_i, error_i, direction_i, outcome_i])
        v_i = W_v(value_i)                        # episode value (e.g. z_memory)
        a_i = softmax(q^T k_i / sqrt(d))
        M   = sum_i a_i * v_i

    The outcome embedding is part of the key, so attention weights depend
    on *what happened* in each remembered episode (SUCCESS/FAILURE/
    RECOVERY), not just on surface similarity.
    """

    def __init__(self, context_dim: int = 256, error_dim: int = 8,
                 direction_dim: int = 16, outcome_dim: int = 8,
                 value_dim: int = 32, key_dim: int = 64,
                 memory_dim: int = 32):
        super().__init__()
        self.context_dim = context_dim
        self.error_dim = error_dim
        self.direction_dim = direction_dim
        self.outcome_dim = outcome_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_dim = memory_dim

        key_in = context_dim + error_dim + direction_dim + outcome_dim
        self.query_proj = nn.Linear(context_dim, key_dim)
        self.key_proj = nn.Linear(key_in, key_dim)
        self.value_proj = nn.Linear(memory_dim, value_dim)
        self.outcome_embedding = OutcomeEmbedding(3, outcome_dim)
        self.scale = 1.0 / (key_dim ** 0.5)

    # ------------------------------------------------------------------ #
    def forward(self,
                query_context: torch.Tensor,
                episode_contexts: Optional[torch.Tensor] = None,
                episode_errors: Optional[torch.Tensor] = None,
                episode_directions: Optional[torch.Tensor] = None,
                episode_values: Optional[torch.Tensor] = None,
                episode_outcomes: Any = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention summary over n retrieved episodes.

        Args:
            query_context:    (context_dim,) or (B, context_dim)
            episode_contexts: (B, n, context_dim) or (n, context_dim) or None
            episode_errors:   (B, n, error_dim)  or (n, error_dim)  or None
            episode_directions: (B, n, direction_dim) or (n, direction_dim) or None
            episode_values:   (B, n, memory_dim) or (n, memory_dim) — the
                              value to aggregate (e.g. each episode's z_memory)
            episode_outcomes: list/array of n labels, or (B, n) labels

        Returns:
            M: (B, value_dim) attention-weighted summary
            attn: (B, n) attention weights
        """
        q_raw = _as_batch(query_context, 1)                    # (Bq, context_dim)

        ctx = _as_batch(episode_contexts, 2) if episode_contexts is not None \
            else torch.zeros(q_raw.shape[0], 1, self.context_dim,
                             device=q_raw.device)
        err = _as_batch(episode_errors, 2) if episode_errors is not None \
            else torch.zeros(q_raw.shape[0], 1, self.error_dim,
                             device=q_raw.device)
        ddir = _as_batch(episode_directions, 2) if episode_directions is not None \
            else torch.zeros(q_raw.shape[0], 1, self.direction_dim,
                             device=q_raw.device)
        vals = _as_batch(episode_values, 2) if episode_values is not None \
            else torch.zeros(q_raw.shape[0], 1, self.memory_dim,
                             device=q_raw.device)

        # broadcast batch dimension (single query / single episode batch)
        B = max(q_raw.shape[0], ctx.shape[0], err.shape[0],
                ddir.shape[0], vals.shape[0])

        def _broadcast_batch(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] == 1 and B > 1:
                return x.expand(B, -1, -1)
            return x

        q_raw = _broadcast_batch(q_raw.unsqueeze(1)).squeeze(1)  # (B, context_dim)
        ctx = _broadcast_batch(ctx)
        err = _broadcast_batch(err)
        ddir = _broadcast_batch(ddir)
        vals = _broadcast_batch(vals)

        q = self.query_proj(q_raw)                               # (B, key_dim)
        n = max(ctx.shape[1], err.shape[1], ddir.shape[1], vals.shape[1])

        def _pad(x: torch.Tensor, dim: int) -> torch.Tensor:
            if x.shape[1] < n:
                pad = torch.zeros(B, n - x.shape[1], dim, device=x.device)
                x = torch.cat([x, pad], dim=1)
            return x

        ctx = _pad(ctx, self.context_dim)
        err = _pad(err, self.error_dim)
        ddir = _pad(ddir, self.direction_dim)
        vals = _pad(vals, self.memory_dim)

        # outcome embedding into the key
        if episode_outcomes is not None:
            emb = self.outcome_embedding(episode_outcomes).to(q.device)
            if emb.dim() == 2 and emb.shape[0] != B * n:
                emb = emb.unsqueeze(0)                   # (1, n, outcome_dim)
            emb = emb.view(B, -1, self.outcome_dim)      # (B, n, outcome_dim)
            emb = _pad(emb, self.outcome_dim)
        else:
            emb = torch.zeros(B, n, self.outcome_dim, device=q.device)

        keys_in = torch.cat([ctx, err, ddir, emb], dim=-1)  # (B, n, key_in)
        keys = self.key_proj(keys_in)                        # (B, n, key_dim)

        scores = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1) * self.scale
        attn = F.softmax(scores, dim=-1)                     # (B, n)

        v = self.value_proj(vals)                            # (B, n, value_dim)
        M = torch.bmm(attn.unsqueeze(1), v).squeeze(1)       # (B, value_dim)
        return M, attn

    # ------------------------------------------------------------------ #
    def forward_single(self, query_context: torch.Tensor,
                       episode_contexts=None, episode_errors=None,
                       episode_directions=None, episode_values=None,
                       episode_outcomes=None,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-query convenience: returns (value_dim,), (n,)."""
        M, attn = self.forward(
            query_context, episode_contexts, episode_errors,
            episode_directions, episode_values, episode_outcomes)
        return M.squeeze(0), attn.squeeze(0)


def _as_batch(x: Optional[torch.Tensor], ndim: int) -> torch.Tensor:
    """Normalize x to include a batch dim.

    ``ndim`` is the number of non-batch trailing dims:
      * ``x.dim() == ndim``     -> no batch dim yet -> unsqueeze(0)
      * ``x.dim() == ndim + 1`` -> already batched  -> keep

    E.g. ndim=1: (d,) -> (1, d); (B, d) stays.  ndim=2: (n, d) -> (1, n, d);
    (B, n, d) stays.
    """
    if x is None:
        raise ValueError("_as_batch called with None")
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == ndim:
        x = x.unsqueeze(0)                     # add batch dim
    elif x.dim() == ndim + 1:
        pass
    else:
        raise ValueError(f"Expected {ndim}D or {ndim + 1}D, got {x.dim()}D")
    return x.float()
