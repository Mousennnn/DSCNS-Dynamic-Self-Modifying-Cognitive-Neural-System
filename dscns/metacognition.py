"""Meta-cognitive control layer (report section 6).

MetaCognitiveState monitors network competence / uncertainty / knowledge
coverage / structure efficiency; MetaCognitiveController turns the verified
knowledge into decisions (internalize / store / defer / reject), adapts
learning rates, and performs active experience selection driven by
information gain, uncertainty, relevance and cost.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .utils import entropy, top_k_indices


class MetaCognitiveState:
    """System self-monitoring state vector C_t."""

    def __init__(self, domains: List[str]):
        self.domains = domains
        self.network_competence: Dict[str, float] = {}
        self.network_uncertainty: Dict[str, float] = {}
        self.knowledge_coverage: Dict[str, float] = {d: 0.0 for d in domains}
        self.structure_efficiency: float = 0.0
        self.learning_progress: List[float] = []
        self.redundancy: float = 0.0
        self.eval_history: List[Dict[str, float]] = []

    def update(self, networks: Dict[str, Any], recent_experiences: List[Any],
               metrics: Optional[Dict[str, float]] = None) -> None:
        for net_id, network in networks.items():
            # competence from recent probe performance
            hist = network.performance_history
            self.network_competence[net_id] = hist[-1] if hist else 0.0
            self.network_uncertainty[net_id] = float(
                np.clip(1.0 - self.network_competence[net_id], 0.0, 1.0)
            )
        # knowledge coverage per domain (episodic memory density)
        episodes = []
        if networks:
            first = next(iter(networks.values()))
            episodes = first.memory.episodic.episodes
        for d in self.domains:
            n_ep = sum(1 for e in episodes if e["context"] == d)
            self.knowledge_coverage[d] = float(np.clip(n_ep / 200.0, 0.0, 1.0))
        if metrics:
            self.learning_progress.append(metrics.get("overall", 0.0))
            self.eval_history.append(metrics)

    def weak_domains(self, threshold: float = 0.35) -> List[str]:
        return [d for d, c in self.knowledge_coverage.items() if c < threshold]


class MetaCognitiveController:
    """Meta-cognitive decision maker."""

    def __init__(self, acceptance_threshold: float = 0.45,
                 store_threshold: float = 0.30,
                 progress_window: int = 10,
                 progress_threshold: float = 0.001):
        self.acceptance_threshold = acceptance_threshold
        self.store_threshold = store_threshold
        self.progress_window = progress_window
        self.progress_threshold = progress_threshold
        self.decision_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def decide(self, candidate: Any, evaluations: Dict[str, Dict[str, float]],
               verified: Any, system_state: MetaCognitiveState,
               memory: Any) -> Dict[str, Any]:
        """Meta decision for one verified knowledge item."""
        # score from verification + meta state
        base_score = verified.confidence if verified.confidence > 0 else 0.4
        avg_R = sum(q.get("R", 0.0) for q in evaluations.values()) / max(len(evaluations), 1)
        avg_N = sum(q.get("N", 0.0) for q in evaluations.values()) / max(len(evaluations), 1)

        # novelty bonus: prefer knowledge that no network holds internally
        novelty_bonus = 0.1 * avg_N

        # coverage penalty: knowledge in already-covered domains is less urgent
        coverage_penalty = 0.05 * system_state.knowledge_coverage.get(candidate.domain, 0.0)

        score = (0.5 * base_score + 0.25 * avg_R + novelty_bonus
                 - coverage_penalty)

        # pick target networks: the most relevant ones (cross-network diversity)
        ranked = sorted(evaluations.items(), key=lambda kv: -kv[1].get("R", 0.0))
        target_networks = [net_id for net_id, _ in ranked[:2] if evaluations[net_id].get("R", 0.0) > 0.15]

        if verified.action == "defer":
            decision = {
                "action": "defer",
                "reason": verified.reason,
                "score": score,
                "target_networks": [],
                "internalization_level": 0.0,
            }
        elif score >= self.acceptance_threshold and target_networks:
            # selective internalization: only for relevant networks
            decision = {
                "action": "internalize",
                "reason": verified.reason,
                "score": score,
                "target_networks": target_networks,
                "internalization_level": float(np.clip(score, 0.2, 1.0)),
            }
        elif score >= self.store_threshold or verified.action == "accept":
            decision = {
                "action": "store",
                "reason": "semantic_storage",
                "score": score,
                "target_networks": [],
                "internalization_level": 0.0,
            }
        else:
            decision = {
                "action": "reject",
                "reason": "low_utility",
                "score": score,
                "target_networks": [],
                "internalization_level": 0.0,
            }
        self.decision_log.append(
            {"knowledge_id": candidate.id, **{k: v for k, v in decision.items()}}
        )
        return decision

    # ------------------------------------------------------------------ #
    def adaptive_learning_rate(self, network: Any, meta_state: MetaCognitiveState) -> float:
        """Adjust learning rate from meta state (report section 6.2)."""
        base_lr = network.base_lr
        uncertainty_factor = 1.0 - meta_state.network_uncertainty.get(network.id, 0.5)
        progress = self._compute_progress(meta_state.learning_progress)
        progress_factor = 1.5 if progress < self.progress_threshold else 1.0
        return base_lr * max(uncertainty_factor, 0.1) * progress_factor

    @staticmethod
    def _compute_progress(history: List[float]) -> float:
        if len(history) < 2:
            return 0.0
        return float(np.mean(np.diff(history[-10:])))

    # ------------------------------------------------------------------ #
    def select_experiences(self, experience_pool: List[Any],
                           budget: int, base_model: Any = None,
                           meta_state: Optional[MetaCognitiveState] = None,
                           learner: Any = None,
                           losses: Optional[np.ndarray] = None) -> List[Any]:
        """Active experience selection (report section 3.6 / 6.2).

        Score(x) = a*IG(x) + b*U(x) + g*R(x) - l*C(x)

        Uncertainty U uses the LEARNER's current losses (adapts to progress);
        relevance R is boosted for weak (low-coverage) domains.
        """
        if len(experience_pool) <= budget:
            return list(experience_pool)
        texts = [x.get("text", "") for x in experience_pool]
        if losses is None:
            losses = np.zeros(len(texts))
        unc = np.clip(1.0 - np.exp(-np.asarray(losses, dtype=np.float32)), 0.0, 1.0)
        weak = set(meta_state.weak_domains()) if meta_state is not None else set()
        scores = []
        for i, x in enumerate(experience_pool):
            # information gain proxy: uncertainty x relevance-to-weak-areas
            R = 1.0 if x.get("domain") in weak else 0.4
            IG = unc[i] * R
            # cost: longer texts cost more compute
            C = min(1.0, len(texts[i]) / 1024.0)
            scores.append(2.0 * IG + 1.0 * unc[i] + 1.0 * R - 0.5 * C)
        idx = top_k_indices(np.asarray(scores), budget)
        return [experience_pool[i] for i in idx]

    # ------------------------------------------------------------------ #
    def should_evolve_structure(self, round_idx: int, config: Any) -> bool:
        if not getattr(config, "evolution_enabled", False):
            return False
        return round_idx > 0 and round_idx % getattr(config, "evolution_every", 5) == 0

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.decision_log:
            counts[d["action"]] = counts.get(d["action"], 0) + 1
        return counts
