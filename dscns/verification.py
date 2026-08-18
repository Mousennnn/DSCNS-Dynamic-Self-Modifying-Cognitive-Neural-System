"""Cross-network verification layer (report section 3.4 / 8.2).

VerificationNetwork aggregates per-network evaluations into a final
confidence, detects conflicts, resolves them with evidence from the memory
systems, and maintains dynamic trust weights per network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeDecision:
    """Outcome of verification + meta decision for one knowledge item."""

    action: str  # 'internalize' | 'store' | 'defer' | 'reject'
    reason: str = ""
    confidence: float = 0.0
    target_networks: Optional[List[str]] = None
    internalization_level: float = 0.0
    revisit_after: int = 0


class VerificationNetwork:
    """Fact-checking / confidence aggregation module (N5)."""

    def __init__(self, trust_weights: Optional[Dict[str, float]] = None,
                 conflict_threshold: float = 0.4,
                 evidence_threshold: float = 0.6,
                 trust_lr: float = 0.05):
        self.trust_weights: Dict[str, float] = trust_weights or {}
        self.conflict_threshold = conflict_threshold
        self.evidence_threshold = evidence_threshold
        self.trust_lr = trust_lr
        self.verify_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def aggregate_confidence(self, evaluations: Dict[str, Dict[str, float]],
                             knowledge_K: Any) -> float:
        """Weighted trust aggregation of per-network confidence.

        w_i = trust_weight(N_i) x relevance_i
        C_final = sum(w_i * C_i) / sum(w_i)
        """
        weighted_sum = 0.0
        weight_total = 0.0
        for net_id, q in evaluations.items():
            w = self.trust_weights.get(net_id, 0.5) * q.get("R", 0.0)
            weighted_sum += w * q.get("C", 0.0)
            weight_total += w
        if weight_total == 0:
            return 0.0
        return weighted_sum / weight_total

    # ------------------------------------------------------------------ #
    def detect_conflict(self, evaluations: Dict[str, Dict[str, float]],
                        threshold: Optional[float] = None) -> bool:
        """Conflict if the relevant networks' confidence spread is too large."""
        threshold = threshold if threshold is not None else self.conflict_threshold
        confidences = [
            q["C"] for q in evaluations.values() if q.get("R", 0.0) > 0.3
        ]
        if len(confidences) < 2:
            return False
        return (max(confidences) - min(confidences)) > threshold

    # ------------------------------------------------------------------ #
    def resolve_conflict(self, knowledge_K: Any, evaluations: Dict[str, Dict[str, float]],
                         memory: Any, revisit_after: int = 100) -> KnowledgeDecision:
        """Conflict resolution strategy (report section 3.4)."""
        # strategy 1: historical evidence from episodic memory
        historical_evidence = memory.episodic.recall(
            {"text": knowledge_K.text, "embedding": knowledge_K.embedding}, k=5
        )
        # strategy 2: related knowledge from semantic memory
        related = memory.semantic.query(
            knowledge_K.text.split()[-1].lower() if knowledge_K.text.split() else "x",
            depth=1,
        )
        if not historical_evidence and not related.get("nodes"):
            return KnowledgeDecision(
                action="defer",
                reason="insufficient_evidence",
                confidence=0.0,
                revisit_after=revisit_after,
            )
        # strategy 4: evidence-based confidence recomputation
        evidence_confidence = self._compute_evidence_confidence(
            knowledge_K, historical_evidence, related
        )
        self._log(knowledge_K, "conflict_resolved", evidence_confidence)
        if evidence_confidence > self.evidence_threshold:
            return KnowledgeDecision(
                action="accept", reason="evidence_supported",
                confidence=evidence_confidence,
            )
        return KnowledgeDecision(
            action="reject", reason="evidence_against", confidence=evidence_confidence
        )

    def _compute_evidence_confidence(self, knowledge_K: Any,
                                     historical_evidence: List[Any],
                                     related: Dict[str, Any]) -> float:
        score = 0.0
        denom = 0.0
        if historical_evidence:
            sims = [e.get("_sim", 0.5) for e in historical_evidence]
            score += 0.6 * (sum(sims) / len(sims))
            denom += 0.6
        if related.get("nodes"):
            confs = [n.get("confidence", 0.5) for n in related["nodes"].values()]
            score += 0.4 * (sum(confs) / len(confs))
            denom += 0.4
        return (score / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------ #
    def update_trust_weight(self, net_id: str, was_correct: bool) -> None:
        """Dynamic trust update based on historical accuracy (report 8.2)."""
        current = self.trust_weights.get(net_id, 0.5)
        if was_correct:
            self.trust_weights[net_id] = min(1.0, current + self.trust_lr)
        else:
            self.trust_weights[net_id] = max(0.1, current - self.trust_lr)

    # ------------------------------------------------------------------ #
    def evaluate_system_decision(self, knowledge_K: Any,
                                 evaluations: Dict[str, Dict[str, float]],
                                 memory: Any,
                                 accept_threshold: float = 0.45) -> KnowledgeDecision:
        """End-to-end verification: aggregate -> conflict -> decide."""
        C_final = self.aggregate_confidence(evaluations, knowledge_K)
        if self.detect_conflict(evaluations):
            conflict = self.resolve_conflict(knowledge_K, evaluations, memory)
            if conflict.action == "accept":
                conflict.confidence = max(conflict.confidence, C_final * 0.8)
                return conflict
            if conflict.action == "defer":
                return conflict
            # 'reject' -> fall through with low confidence
            if conflict.confidence <= self.evidence_threshold:
                return conflict
        # no conflict: acceptance driven by aggregated confidence and relevance
        avg_R = sum(q.get("R", 0.0) for q in evaluations.values()) / max(len(evaluations), 1)
        avg_I = sum(q.get("I", 0.0) for q in evaluations.values()) / max(len(evaluations), 1)
        score = 0.5 * C_final + 0.3 * avg_R + 0.2 * avg_I
        self._log(knowledge_K, "aggregated", score)
        if score >= accept_threshold:
            return KnowledgeDecision(
                action="accept", reason="aggregated_confidence",
                confidence=score,
            )
        return KnowledgeDecision(
            action="store", reason="below_acceptance", confidence=score
        )

    # ------------------------------------------------------------------ #
    def _log(self, knowledge_K: Any, stage: str, score: float) -> None:
        self.verify_log.append(
            {"knowledge_id": getattr(knowledge_K, "id", None), "stage": stage,
             "score": float(score)}
        )

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.verify_log:
            counts[entry["stage"]] = counts.get(entry["stage"], 0) + 1
        return counts
