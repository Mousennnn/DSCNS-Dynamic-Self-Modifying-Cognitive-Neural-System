"""Shared numerical helpers for DSCNS."""
from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix (n_a, n_b)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_n = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a @ b.T) / (a_n @ b_n.T)


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy (nats) of a probability vector."""
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log(p) for p in probs)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


def top_k_indices(scores: np.ndarray, k: int) -> List[int]:
    """Indices of the k largest scores."""
    k = min(k, len(scores))
    return list(np.argsort(-np.asarray(scores))[:k])


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def loss_to_confidence(loss: float, scale: float = 1.0) -> float:
    """Map a CE loss to a [0,1] confidence: lower loss -> higher confidence."""
    return float(np.clip(1.0 / (1.0 + math.exp(loss / scale)), 0.0, 1.0))


def loss_to_uncertainty(loss: float) -> float:
    """Map a CE loss to a [0,1] uncertainty proxy: higher loss -> more uncertain."""
    return float(np.clip(1.0 - math.exp(-loss), 0.0, 1.0))
