"""DSCNSSystem: the full cognitive loop (report section 7.3 / 11).

    experience -> parse -> multi-network observation -> independent
    evaluation -> cross-network verification -> meta decision ->
    selective progressive internalization -> memory update ->
    structure evolution check -> new system state.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .communication import MessageType, NetworkCommunicationBus, NetworkMessage
from .experience import CandidateParser, ExperienceBuffer
from .internalization import InternalizationController
from .memory import MemorySystem
from .metacognition import MetaCognitiveController, MetaCognitiveState
from .networks import (CognitiveNetwork, LanguageNetwork, LogicNetwork,
                       MathNetwork, VerificationNetwork, WorldKnowledgeNetwork)
from .verification import VerificationNetwork as Verifier


class DSCNSSystem:
    """Dynamic Self-Modifying Cognitive Network System."""

    def __init__(self, base_model: Any, config: Any,
                 domain_exemplars: Dict[str, List[str]],
                 probe_sets: Dict[str, List[str]],
                 networks_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 trust_initial: Optional[Dict[str, float]] = None,
                 seed: int = 42):
        self.base_model = base_model
        self.config = config
        self.rng = np.random.RandomState(seed)

        # memory systems (shared)
        self.memory = MemorySystem()

        # domain embeddings from frozen-base exemplar embeddings
        self.domain_exemplars = domain_exemplars
        self.probe_sets = probe_sets
        self.domain_embeddings = self._compute_domain_embeddings(domain_exemplars)

        # cognitive networks
        self.networks: Dict[str, CognitiveNetwork] = {}
        self._initialize_networks(networks_config or {})

        # verification / meta / internalization
        self.verifier = Verifier(
            trust_weights=dict(trust_initial or {}),
            conflict_threshold=getattr(config, "conflict_threshold", 0.4),
            trust_lr=getattr(config, "trust_lr", 0.05),
        )
        self.meta_controller = MetaCognitiveController(
            acceptance_threshold=getattr(config, "acceptance_threshold", 0.45),
        )
        self.meta_state = MetaCognitiveState(domains=list(domain_exemplars.keys()))
        self.internalizer = InternalizationController(
            tolerance=getattr(config, "internalization_tolerance", 0.02),
            max_alpha=getattr(config, "max_alpha", 0.1),
            steps=getattr(config, "internalization_steps", 5),
        )

        # experience management
        self.experience_buffer = ExperienceBuffer(
            capacity=getattr(config, "buffer_capacity", 10000)
        )
        self.parser = CandidateParser(
            base_model, max_len=getattr(config, "max_len", 256),
            batch_size=getattr(config, "parse_batch", 8),
        )

        # communication
        self.bus = NetworkCommunicationBus(networks=self.networks)

        self.round_idx = 0
        self.eval_history: List[Dict[str, Any]] = []
        self.connections: Dict[tuple, float] = {}
        self.eval_sets: Dict[str, List[str]] = {}
        self._round_trials_used = 0

    # ------------------------------------------------------------------ #
    # initialization
    # ------------------------------------------------------------------ #
    def _compute_domain_embeddings(self, exemplars: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        embs = {}
        for domain, texts in exemplars.items():
            if texts:
                e = self.base_model.embed(texts[:64], batch_size=8)
                embs[domain] = np.mean(e, axis=0)
            else:
                embs[domain] = None
        return embs

    def _initialize_networks(self, networks_config: Dict[str, Dict[str, Any]]) -> None:
        cfg = self.config
        n = getattr(cfg, "num_networks", 5)
        if n <= 1:
            # Exp1: one universal learner network
            self._add_network("N1", "Learner", "general",
                              self._universal_embedding())
            return
        specs = [
            ("N1", "WorldKnowledge", "general", "world", "general"),
            ("N2", "Math", "math", "math", "math"),
            ("N3", "Logic", "logic", "logic", "logic"),
            ("N4", "Language", "language", "language", "code"),
            ("N5", "Verification", "verification", "verification", "science"),
        ]
        domain_key = {"world": "general", "math": "math", "logic": "logic",
                      "language": "code", "verification": "science"}
        for net_id, cls_name, domain, key, data_domain in specs:
            emb = self.domain_embeddings.get(domain_key[key])
            self._add_network(net_id, cls_name, domain, emb, data_domain)

    def _universal_embedding(self) -> np.ndarray:
        embs = [e for e in self.domain_embeddings.values() if e is not None]
        if not embs:
            return None
        return np.mean(np.stack(embs), axis=0)

    def _add_network(self, net_id: str, name: str, domain: str,
                     domain_embedding: Optional[np.ndarray],
                     data_domain: Optional[str] = None) -> None:
        self.base_model.add_adapter(net_id)
        cls = {
            "WorldKnowledge": WorldKnowledgeNetwork,
            "Math": MathNetwork,
            "Logic": LogicNetwork,
            "Language": LanguageNetwork,
            "Verification": VerificationNetwork,
        }.get(name, CognitiveNetwork)
        net = cls(
            net_id=net_id, name=name, domain=domain,
            peft_model=self.base_model.peft_model,
            memory=self.memory,
            domain_embedding=domain_embedding,
            base_lr=getattr(self.config, "lora_lr", 5e-4),
        )
        net.data_domain = data_domain
        self.networks[net_id] = net
        net.set_trainable(False)

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def process_experiences(self, experiences: List[Dict[str, Any]],
                            with_query: bool = True) -> Dict[str, Any]:
        """Full pipeline for a batch of experiences (one round step)."""
        # 1. parse candidates
        candidates = self.parser.parse(experiences)
        self.experience_buffer.extend(experiences)

        # 2. broadcast observation (Level 1: existence-level cognition)
        for cand in candidates:
            self.bus.send_sync(NetworkMessage(
                "system", "broadcast", MessageType.BROADCAST,
                {"candidate": cand},
            ))

        # 3. independent evaluation + 4. cross-network verification
        decisions = {}
        internalize_groups: Dict[str, List[Any]] = {}
        for cand in candidates:
            evaluations = {}
            for net_id, net in self.networks.items():
                evaluations[net_id] = net.evaluate(cand)
                net.observe(cand)
                self.bus.send_sync(NetworkMessage(
                    net_id, "verifier", MessageType.CONFIDENCE, evaluations[net_id],
                ))

            # inter-network query between the two most relevant networks
            if with_query and len(self.networks) > 1:
                ranked = sorted(evaluations.items(), key=lambda kv: -kv[1].get("R", 0.0))
                q_net, r_net = ranked[0][0], ranked[1][0]
                self.bus.send_sync(NetworkMessage(
                    q_net, r_net, MessageType.QUERY, {"knowledge_id": cand.id},
                ))
                response_conf = evaluations[r_net].get("C", 0.5)
                evaluations[q_net]["C"] = float(np.clip(
                    0.8 * evaluations[q_net]["C"] + 0.2 * response_conf, 0.0, 1.0
                ))

            verified = self.verifier.evaluate_system_decision(cand, evaluations, self.memory)
            if self.verifier.detect_conflict(evaluations):
                self.bus.send_sync(NetworkMessage(
                    "verifier", "broadcast", MessageType.CONFLICT,
                    {"knowledge_id": cand.id},
                ))

            # 5. meta-cognitive decision
            decision = self.meta_controller.decide(
                cand, evaluations, verified, self.meta_state, self.memory
            )
            decisions[cand.id] = decision
            if decision["action"] == "internalize":
                for net_id in decision["target_networks"]:
                    internalize_groups.setdefault(net_id, []).append(cand)
            elif decision["action"] == "store":
                cand.states = {net_id: 2 for net_id in self.networks}
                self.memory.store_semantic(cand, list(self.networks.keys()),
                                           confidence=decision["score"])
            cand.internalization = {
                net_id: lvl for net_id, lvl in
                ((k, decision["internalization_level"])
                 for k in decision["target_networks"])
            }
            # episodic storage (all observed experiences, traceable)
            self.memory.store_episode(
                {"text": cand.text, "embedding": cand.embedding,
                 "domain": cand.domain, "source": cand.source,
                 "knowledge_id": cand.id,
                 "activated_networks": list(self.networks.keys())},
                context=cand.domain,
            )

        # 6. selective progressive internalization
        internalization_stats = self._internalize_groups(internalize_groups)
        self._round_trials_used = 0

        # 7. structural evolution check
        evolve_stats = {}
        if getattr(self.config, "evolution_enabled", False):
            evolve_stats = self.evolve_structure()

        self.round_idx += 1
        return {
            "n_candidates": len(candidates),
            "decisions": decisions,
            "internalization": internalization_stats,
            "evolution": evolve_stats,
            "memory": self.memory.snapshot(),
        }

    # ------------------------------------------------------------------ #
    def _internalize_groups(self, groups: Dict[str, List[Any]]) -> Dict[str, Any]:
        stats = {}
        budget = getattr(self.config, "max_grad_steps_per_round", 8)
        # order groups by aggregate relevance of their knowledge
        ordered = sorted(groups.items(),
                         key=lambda kv: -np.mean([c.internalization.get(kv[0], 0.0)
                                                  for c in kv[1]]))
        for net_id, cands in ordered:
            if self._round_trials_used >= budget:
                # budget exhausted -> store remaining knowledge as callable
                for c in cands:
                    for nid in self.networks:
                        self.networks[nid].store_as_callable(c)
                    self.memory.store_semantic(c, list(self.networks.keys()),
                                               confidence=0.3)
                continue
            net = self.networks[net_id]
            target_level = float(np.mean([c.internalization.get(net_id, 0.5)
                                          for c in cands]))
            probe_perf_before = self.regression_test(net)
            remaining = budget - self._round_trials_used
            result = self.internalizer.internalize(
                net, cands, target_level, self.base_model.tokenizer,
                lambda n: self.regression_test(n),
                max_steps=remaining,
            )
            self._round_trials_used += result.trials
            probe_perf_after = self.regression_test(net)
            net.performance_history.append(probe_perf_after)
            net.baseline_performance = probe_perf_after
            # dynamic trust update: correct if probe performance improved
            self.verifier.update_trust_weight(net_id,
                                              probe_perf_after >= probe_perf_before)
            for c in cands:
                c.internalization[net_id] = result.final_level
            self.bus.send_sync(NetworkMessage(
                net_id, "broadcast", MessageType.UPDATE_NOTIFY,
                {"n_knowledge": len(cands), "result": result.success},
            ))
            stats[net_id] = {
                "success": result.success,
                "final_level": result.final_level,
                "trials": result.trials,
                "perf_before": probe_perf_before,
                "perf_after": probe_perf_after,
                "stop_reason": result.stop_reason,
            }
        return stats

    # ------------------------------------------------------------------ #
    def regression_test(self, network: CognitiveNetwork) -> float:
        """Probe performance across all domains (used as regression test)."""
        perfs = []
        for texts in self.probe_sets.values():
            if texts:
                perfs.append(network.evaluate_texts(
                    texts[: getattr(self.config, "probe_size", 16)],
                    self.base_model.tokenizer,
                    batch_size=getattr(self.config, "probe_batch", 8),
                ))
        return float(np.mean(perfs)) if perfs else 0.0

    # ------------------------------------------------------------------ #
    def evaluate_domains(self, eval_sets: Dict[str, List[str]],
                         eval_size: int = 64) -> Dict[str, Dict[str, float]]:
        """Per-domain performance: best network per domain + per-network values."""
        results: Dict[str, Dict[str, float]] = {}
        for domain, texts in eval_sets.items():
            texts = texts[:eval_size]
            net_perfs = {}
            for net_id, net in self.networks.items():
                if texts:
                    net_perfs[net_id] = net.evaluate_texts(
                        texts, self.base_model.tokenizer,
                        batch_size=getattr(self.config, "eval_batch", 16),
                    )
            results[domain] = net_perfs
        return results

    def best_domain_performance(self, eval_sets: Dict[str, List[str]],
                                eval_size: int = 64) -> Dict[str, float]:
        results = self.evaluate_domains(eval_sets, eval_size)
        best = {}
        for domain, net_perfs in results.items():
            best[domain] = max(net_perfs.values()) if net_perfs else 0.0
        return best

    # ------------------------------------------------------------------ #
    def evolve_structure(self) -> Dict[str, Any]:
        """Run split / merge / connect checks (report section 4)."""
        from .evolution import StructureEvolver

        min_round = getattr(self.config, "evolution_min_round", 6)
        if self.round_idx < min_round:
            # stabilization period: networks need time to specialize first
            self.evolver = StructureEvolver()
            return {"evolutions": [], "n_networks": len(self.networks),
                    "connections": len(self.connections), "changed": False,
                    "reason": f"before_min_round_{min_round}"}
        evolver = StructureEvolver(
            diversity_threshold=getattr(self.config, "split_diversity_threshold", 0.8),
            overlap_threshold=getattr(self.config, "merge_overlap_threshold", 0.97),
            co_activation_threshold=getattr(self.config, "merge_co_activation_threshold", 8),
            similarity_threshold=getattr(self.config, "merge_similarity_threshold", 0.9),
        )
        self.evolver = evolver
        nets = dict(self.networks)
        perf_by_domain = self.best_domain_performance(self.eval_sets)
        # splits (at most one per round)
        for net_id in list(nets.keys()):
            net = nets[net_id]
            if evolver.should_split(net, perf_by_domain, self.round_idx):
                nets = evolver.split_network(
                    net, nets, self.round_idx,
                    peft_model=self.base_model.peft_model,
                    lora_kwargs={"r": getattr(self.config, "lora_r", 16),
                                 "lora_alpha": getattr(self.config, "lora_alpha", 32),
                                 "lora_dropout": getattr(self.config, "lora_dropout", 0.1)},
                    tmp_dir=os.path.join("models", "tmp_adapters"),
                )
                break  # at most one split per round (stabilization)
        # merges (at most one per round)
        merged = False
        ids = list(nets.keys())
        for i in range(len(ids)):
            if merged:
                break
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                co = self.bus.get_co_activation_matrix().get((a, b), 0)
                if evolver.should_merge(nets[a], nets[b], co):
                    nets = evolver.merge_networks(nets[a], nets[b], nets, self.round_idx)
                    merged = True
                    break
        # connections
        self.connections = evolver.update_connections(
            nets, self.bus.get_co_activation_matrix(), round_idx=self.round_idx
        )
        changed = set(nets.keys()) != set(self.networks.keys())
        self.networks = nets
        self.bus.networks = nets
        return {"evolutions": evolver.evolution_log[-10:],
                "n_networks": len(nets),
                "connections": len(self.connections),
                "changed": changed}

    # ------------------------------------------------------------------ #
    def meta_update(self, metrics: Optional[Dict[str, float]] = None) -> None:
        self.meta_state.update(self.networks, [], metrics)
        for net in self.networks.values():
            net.lr = self.meta_controller.adaptive_learning_rate(net, self.meta_state)

    def set_eval_sets(self, eval_sets: Dict[str, List[str]]) -> None:
        """Register held-out eval sets (used by structure evolution)."""
        self.eval_sets = eval_sets
