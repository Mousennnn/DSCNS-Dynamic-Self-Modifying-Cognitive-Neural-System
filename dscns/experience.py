"""Experience management layer (report section 2.1 / 3.1).

Implements the ExperienceBuffer and CandidateParser: raw experiences from the
environment are buffered, parsed into candidate knowledge items (each with an
embedding, domain tag, source reliability and uncertainty proxy), and then
broadcast for multi-network observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .memory import knowledge_id


@dataclass
class Candidate:
    """A parsed candidate knowledge item K."""

    id: str
    text: str
    domain: str
    embedding: Optional[np.ndarray] = None
    source: str = "unknown"
    source_reliability: float = 0.5
    uncertainty: float = 0.5  # base-model uncertainty proxy on this item
    loss: float = 0.0
    internalization: Dict[str, float] = field(default_factory=dict)  # I_ij
    states: Dict[str, int] = field(default_factory=dict)  # Level 0..3 per network
    meta: Dict[str, Any] = field(default_factory=dict)


class ExperienceBuffer:
    """Buffers raw experiences; supports windowed sampling.

    Principle P1: receiving information != learning.  The buffer is fully
    decoupled from parameter updates.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def push(self, experience: Dict[str, Any]) -> None:
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def extend(self, experiences: List[Dict[str, Any]]) -> None:
        for e in experiences:
            self.push(e)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> List[Dict[str, Any]]:
        rng = rng or np.random
        if len(self.buffer) <= n:
            return list(self.buffer)
        idx = rng.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class CandidateParser:
    """Parses raw experiences into candidate knowledge items.

    Each candidate receives:
    - an embedding from the (frozen) base model,
    - a source-reliability estimate,
    - an uncertainty proxy derived from the base-model loss.
    """

    def __init__(self, base_model: Any, max_len: int = 256, batch_size: int = 8):
        self.base_model = base_model
        self.max_len = max_len
        self.batch_size = batch_size

    def parse(self, experiences: List[Dict[str, Any]]) -> List[Candidate]:
        if not experiences:
            return []
        texts = [e["text"] for e in experiences]
        # batch-embed and batch-loss in one pass
        embs, losses = self.base_model.embed_and_loss(
            texts, max_len=self.max_len, batch_size=self.batch_size
        )
        candidates = []
        for i, exp in enumerate(experiences):
            src = exp.get("source", "dataset")
            rel = exp.get("source_reliability", 0.8)
            if "reliability" in exp:
                rel = exp["reliability"]
            loss = float(losses[i])
            cand = Candidate(
                id=knowledge_id(texts[i]),
                text=texts[i],
                domain=exp.get("domain", "general"),
                embedding=np.asarray(embs[i], dtype=np.float32),
                source=src,
                source_reliability=float(rel),
                uncertainty=float(np.clip(1.0 - np.exp(-loss), 0.0, 1.0)),
                loss=loss,
                meta=dict(exp),
            )
            candidates.append(cand)
        return candidates
