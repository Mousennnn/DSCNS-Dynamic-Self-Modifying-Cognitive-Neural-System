"""Three-layer memory architecture (report section 5).

- EpisodicMemory: time-ordered raw experiences with embedding-based recall.
- SemanticMemory:  knowledge graph of abstracted concepts with confidence
  and per-network internalization levels (I_ij).
- ProceduralMemory: successful action sequences per task type.
- MemorySystem:   aggregate facade used by the cognitive networks.
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import cosine_matrix


def knowledge_id(text: str) -> str:
    """Stable id for a candidate knowledge item."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


class EpisodicMemory:
    """Episodic memory: verbatim experiences with retrieval."""

    def __init__(self, capacity: int = 5000):
        self.episodes: List[Dict[str, Any]] = []
        self.capacity = capacity

    def store(self, experience: Dict[str, Any], timestamp: float = None,
              context: str = "") -> None:
        episode = {
            "content": experience.get("text", ""),
            "embedding": experience.get("embedding"),
            "time": timestamp if timestamp is not None else time.time(),
            "context": context or experience.get("domain", ""),
            "source": experience.get("source", "unknown"),
            "networks_activated": experience.get("activated_networks", []),
            "knowledge_id": experience.get("knowledge_id",
                                           knowledge_id(experience.get("text", ""))),
        }
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def recall(self, query: Dict[str, Any], k: int = 10,
               by_embedding: bool = True) -> List[Dict[str, Any]]:
        """Retrieve top-k similar episodes.

        ``query`` may carry an 'embedding' (vector recall) or 'text'/'domain'
        (keyword + domain filtering fallback).
        """
        q_emb = query.get("embedding")
        if by_embedding and q_emb is not None and self.episodes:
            embs = np.stack(
                [e["embedding"] for e in self.episodes
                 if e.get("embedding") is not None] or [np.zeros_like(q_emb)],
                axis=0,
            )
            if embs.ndim == 2 and embs.shape[0] > 0:
                sims = cosine_matrix(np.asarray(q_emb)[None, :], embs)[0]
                order = np.argsort(-sims)[: min(k, len(sims))]
                hits = []
                all_eps = [e for e in self.episodes if e.get("embedding") is not None]
                for idx in order:
                    hit = dict(all_eps[int(idx)])
                    hit["_sim"] = float(sims[int(idx)])
                    hits.append(hit)
                return hits
        # fallback: domain + text overlap filtering
        q_domain = query.get("domain", "")
        scored = []
        for e in self.episodes:
            s = 0.0
            if q_domain and e["context"] == q_domain:
                s += 1.0
            for w in str(query.get("text", "")).split()[:20]:
                if w.lower() in e["content"].lower():
                    s += 0.05
            scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [dict(e) for _, e in scored[:k]]

    def __len__(self) -> int:
        return len(self.episodes)


class SemanticMemory:
    """Semantic memory: a lightweight knowledge graph.

    Nodes are concept keys (frequent tokens/keywords of knowledge items);
    edges carry relation weights.  Each node records its confidence and the
    per-network internalization level I_ij (report section 3.2).
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self.node_knowledge: Dict[str, List[str]] = defaultdict(list)

    def store(self, concept: str, relations: Optional[List[Tuple[str, str]]] = None,
              confidence: float = 0.5, knowledge_item: Optional[str] = None) -> str:
        node = self.nodes.setdefault(
            concept,
            {
                "concept": concept,
                "confidence": confidence,
                "internalization_level": {},  # network_id -> I_ij
                "importance": confidence,
                "created": time.time(),
            },
        )
        node["confidence"] = 0.7 * node["confidence"] + 0.3 * confidence
        if knowledge_item is not None:
            self.node_knowledge[concept].append(knowledge_item)
        for rel, weight in (relations or []):
            self.edges[(concept, rel)] += weight
            self.edges.setdefault((rel, concept), 0.0)
        return concept

    def update_internalization(self, concept: str, network_id: str,
                               level: float) -> None:
        node = self.nodes.get(concept)
        if node is not None:
            node["internalization_level"][network_id] = float(level)

    def query(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Query a concept and its neighbors up to ``depth`` hops."""
        if concept not in self.nodes:
            return {}
        visited = {concept}
        frontier = [concept]
        result = {"nodes": {concept: self.nodes[concept]},
                  "edges": []}
        for _ in range(depth):
            nxt = []
            for c in frontier:
                for (a, b), w in self.edges.items():
                    if a == c and b not in visited:
                        visited.add(b)
                        if b in self.nodes:
                            result["nodes"][b] = self.nodes[b]
                            result["edges"].append((a, b, w))
                        nxt.append(b)
            frontier = nxt
            if not frontier:
                break
        return result

    def coverage(self) -> int:
        return len(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


class ProceduralMemory:
    """Procedural memory: successful action sequences per task type."""

    def __init__(self):
        self.procedures: Dict[str, List[Dict[str, Any]]] = {}

    def store(self, task_type: str, successful_procedure: Any) -> None:
        if task_type not in self.procedures:
            self.procedures[task_type] = []
        self.procedures[task_type].append(
            {"steps": successful_procedure, "success_rate": 1.0, "usage_count": 1}
        )
        # keep only best 8 procedures per task type
        self.procedures[task_type].sort(key=lambda p: -p["success_rate"])
        self.procedures[task_type] = self.procedures[task_type][:8]

    def retrieve(self, task_type: str) -> Optional[Dict[str, Any]]:
        if task_type in self.procedures and self.procedures[task_type]:
            best = max(self.procedures[task_type], key=lambda p: p["success_rate"])
            best["usage_count"] += 1
            return best
        return None

    def record_outcome(self, task_type: str, was_successful: bool) -> None:
        if task_type in self.procedures and self.procedures[task_type]:
            p = self.procedures[task_type][-1]
            n = p["usage_count"]
            p["success_rate"] = (p["success_rate"] * n + (1.0 if was_successful else 0.0)) / (n + 1)


class MemorySystem:
    """Aggregate memory facade (M_t = (M_t^ep, M_t^sem, M_t^proc))."""

    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()

    def store_episode(self, experience: Dict[str, Any], context: str = "") -> None:
        self.episodic.store(experience, context=context)

    def store_semantic(self, candidate: Any, network_ids: List[str],
                       confidence: float = 0.5) -> None:
        """Store an abstracted knowledge item into the knowledge graph."""
        concepts = self._extract_concepts(candidate.text)
        relations = [(c, 1.0) for c in concepts[1:]]
        for concept in concepts[:3]:
            self.semantic.store(
                concept,
                relations=relations,
                confidence=confidence,
                knowledge_item=candidate.id,
            )
            for net_id in network_ids:
                level = candidate.internalization.get(net_id, 0.0)
                self.semantic.update_internalization(concept, net_id, level)

    def store_procedure(self, task_type: str, steps: Any) -> None:
        self.procedural.store(task_type, steps)

    @staticmethod
    def _extract_concepts(text: str, max_concepts: int = 3) -> List[str]:
        """Very light concept extraction: capitalized / frequent tokens."""
        import re

        words = re.findall(r"[A-Za-z][A-Za-z\-']{3,}", text or "")
        freq: Dict[str, int] = defaultdict(int)
        for w in words:
            freq[w.lower()] += 1
        caps = [w for w in words if w[0].isupper() and w.lower() not in
                {"The", "A", "An", "This", "That", "These", "Those", "It",
                 "They", "We", "I", "He", "She", "In", "On", "At", "For",
                 "With", "From", "By", "To", "Of", "And", "Or", "But"}]
        concepts = []
        seen = set()
        for w in caps + sorted(freq, key=lambda x: -freq[x]):
            key = w.lower()
            if key not in seen and len(key) > 3:
                seen.add(key)
                concepts.append(key)
            if len(concepts) >= max_concepts:
                break
        return concepts or ["unknown"]

    def snapshot(self) -> Dict[str, int]:
        return {
            "episodic": len(self.episodic),
            "semantic_nodes": len(self.semantic),
            "procedural_types": len(self.procedural.procedures),
        }
