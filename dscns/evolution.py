"""Network structure evolution (report section 4).

Implements specialization scoring, split / merge triggers and strategies,
and dynamic connection establishment via the co-activation matrix and
information flow.  All operations are conservative: they run only when the
corresponding thresholds are exceeded and are followed by a stabilization
period (report section 14.1 risk mitigation).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import cosine_similarity, entropy


def _lora_config(peft_model: Any) -> Any:
    """A fresh LoraConfig templated on an existing adapter."""
    from peft import LoraConfig

    first = next(iter(peft_model.peft_config.values()))
    return LoraConfig(
        r=first.r, lora_alpha=first.lora_alpha, lora_dropout=first.lora_dropout,
        target_modules=first.target_modules, bias=first.bias,
        task_type=first.task_type,
    )


def _load_adapter_state(peft_model: Any, snap: Dict[str, Any], new_id: str) -> None:
    """Copy a source adapter's weights into a freshly added adapter."""
    import re

    src_ids = [k.split(".")[-2] for k in snap.keys() if "lora" in k]
    src = src_ids[0] if src_ids else None
    if src is None:
        return
    state = peft_model.state_dict()
    for k, v in state.items():
        if f".{new_id}." in k:
            src_key = re.sub(rf"\.{re.escape(new_id)}\.", f".{src}.", k)
            if src_key in snap:
                v.data.copy_(snap[src_key])


class StructureEvolver:
    """Split / merge / connect decisions over the network population."""

    def __init__(self, diversity_threshold: float = 0.8,
                 cluster_threshold: int = 2,
                 negative_transfer_threshold: float = 0.05,
                 overlap_threshold: float = 0.8,
                 co_activation_threshold: int = 3,
                 similarity_threshold: float = 0.9,
                 connect_threshold: float = 0.5,
                 co_act_w: float = 0.5, info_flow_w: float = 0.5,
                 stabilization_rounds: int = 2):
        self.diversity_threshold = diversity_threshold
        self.cluster_threshold = cluster_threshold
        self.negative_transfer_threshold = negative_transfer_threshold
        self.overlap_threshold = overlap_threshold
        self.co_activation_threshold = co_activation_threshold
        self.similarity_threshold = similarity_threshold
        self.connect_threshold = connect_threshold
        self.co_act_w = co_act_w
        self.info_flow_w = info_flow_w
        self.stabilization_rounds = stabilization_rounds
        self.evolution_log: List[Dict[str, Any]] = []
        self._last_change_round = -10

    # ------------------------------------------------------------------ #
    # 4.1 specialization
    # ------------------------------------------------------------------ #
    @staticmethod
    def specialization_score(network: Any, domain: str,
                             performance_by_domain: Dict[str, float]) -> float:
        perf = performance_by_domain.get(domain, 0.0)
        avg = sum(performance_by_domain.values()) / max(len(performance_by_domain), 1)
        activation = network.recent_domains.count(domain) / max(len(network.recent_domains), 1)
        if avg <= 0:
            return 0.0
        return (perf / avg) * activation

    # ------------------------------------------------------------------ #
    # 4.2 split
    # ------------------------------------------------------------------ #
    def should_split(self, network: Any, performance_by_domain: Dict[str, float],
                     round_idx: int) -> bool:
        if round_idx - self._last_change_round < self.stabilization_rounds:
            return False
        task_diversity = network.task_diversity()
        if task_diversity <= self.diversity_threshold:
            return False
        if len(network.accepted_embeddings) < self.cluster_threshold * 4:
            return False
        clusters = self._cluster_count(network.accepted_embeddings)
        if clusters < self.cluster_threshold:
            return False
        neg_transfer = self._detect_negative_transfer(network, performance_by_domain)
        return neg_transfer > self.negative_transfer_threshold

    @staticmethod
    def _cluster_count(embeddings: List[np.ndarray],
                       max_k: int = 4, min_k: int = 2) -> int:
        """Simple silhouette-free cluster count via k-means inertia elbow."""
        from sklearn.cluster import KMeans

        X = np.stack(embeddings)
        n = len(X)
        if n < 4:
            return 1
        best_k, best_score = 1, -1e9
        for k in range(min_k, min(max_k, n) + 1):
            km = KMeans(n_clusters=k, n_init=2, random_state=0).fit(X)
            inertia = km.inertia_
            score = inertia * (n ** 0.5)  # normalized inertia (lower = worse)
            # elbow: relative drop
            if k == min_k:
                prev = score
                continue
            drop = (prev - score) / prev
            if drop > 0.25:
                best_k, best_score = k, drop
            prev = score
        return best_k if best_k > 1 else 1

    @staticmethod
    def _detect_negative_transfer(network: Any,
                                  performance_by_domain: Dict[str, float]) -> float:
        """Negative transfer: primary-domain perf up while others drift down."""
        if not network.performance_history or len(network.performance_history) < 2:
            return 0.0
        # simple proxy: variance of per-domain performance
        vals = list(performance_by_domain.values())
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals) / max(np.mean(vals), 1e-6))

    def split_network(self, network: Any, networks: Dict[str, Any],
                      round_idx: int, peft_model: Any = None,
                      lora_kwargs: Optional[Dict[str, Any]] = None,
                      tmp_dir: str = None) -> Dict[str, Any]:
        """N_i -> [N_i^a, N_i^b] with disjoint task partition.

        Each child receives its own LoRA adapter: if ``peft_model`` is given,
        the parent's adapter weights are copied to two fresh adapters (so the
        children start from the parent's knowledge and then specialize).
        """
        from sklearn.cluster import KMeans

        X = np.stack(network.accepted_embeddings)
        k = min(2, len(X))
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X)
        ids_a = [network.accepted_ids[i] for i in np.where(km.labels_ == 0)[0]]
        ids_b = [network.accepted_ids[i] for i in np.where(km.labels_ == 1)[0]]
        if not ids_a or not ids_b:  # degenerate -> no split
            return networks

        # create fresh adapters initialized from the parent's weights
        # (pure in-memory copy: peft's save_pretrained/load_adapter path
        #  misvalidates local paths as Hub repo ids on Windows)
        if peft_model is not None:
            snap = {k: v.detach().clone()
                    for k, v in peft_model.state_dict().items()
                    if f".{network.id}." in k or k.endswith(f".{network.id}")}
            for new_id in (f"{network.id}a", f"{network.id}b"):
                peft_model.add_adapter(new_id, _lora_config(peft_model))
            _load_adapter_state(peft_model, snap, f"{network.id}a")
            _load_adapter_state(peft_model, snap, f"{network.id}b")

        net_a = self._clone_network(network, f"{network.id}a", "split_a")
        net_b = self._clone_network(network, f"{network.id}b", "split_b")
        # domain embeddings follow the cluster centroids
        net_a.domain_embedding = km.cluster_centers_[0]
        net_b.domain_embedding = km.cluster_centers_[1]
        # knowledge bookkeeping partition
        for cid in ids_a:
            net_a.internalization_level[cid] = network.internalization_level.get(cid, 1.0)
            net_a.knowledge_states[cid] = 3
        for cid in ids_b:
            net_b.internalization_level[cid] = network.internalization_level.get(cid, 1.0)
            net_b.knowledge_states[cid] = 3

        new_nets = dict(networks)
        del new_nets[network.id]
        new_nets[net_a.id] = net_a
        new_nets[net_b.id] = net_b
        self.evolution_log.append(
            {"op": "split", "source": network.id, "into": [net_a.id, net_b.id],
             "round": round_idx}
        )
        self._last_change_round = round_idx
        return new_nets

    @staticmethod
    def _clone_network(network: Any, new_id: str, new_domain: str) -> Any:
        import copy

        clone = copy.copy(network)
        clone.id = new_id
        clone.name = f"{network.name}-{new_id}"
        clone.domain = new_domain
        clone.recent_domains = list(network.recent_domains)
        clone.accepted_embeddings = list(network.accepted_embeddings)
        clone.accepted_ids = list(network.accepted_ids)
        clone.internalization_level = dict(network.internalization_level)
        clone.knowledge_states = dict(network.knowledge_states)
        clone.performance_history = list(network.performance_history)
        clone.inbox = []
        clone.optimizer = None
        return clone

    # ------------------------------------------------------------------ #
    # 4.3 merge
    # ------------------------------------------------------------------ #
    def should_merge(self, net_i: Any, net_j: Any,
                     co_activation: int) -> bool:
        overlap = self._functional_overlap(net_i, net_j)
        rep_sim = self._representation_similarity(net_i, net_j)
        return (overlap > self.overlap_threshold and
                co_activation >= self.co_activation_threshold and
                rep_sim > self.similarity_threshold)

    def _functional_overlap(self, net_i: Any, net_j: Any) -> float:
        if net_i.domain_embedding is None or net_j.domain_embedding is None:
            return 0.0
        return cosine_similarity(net_i.domain_embedding, net_j.domain_embedding)

    @staticmethod
    def _representation_similarity(net_i: Any, net_j: Any) -> float:
        e_i, e_j = net_i.representation_embedding(), net_j.representation_embedding()
        if e_i.size == 1 or e_j.size == 1:
            return 0.0
        return cosine_similarity(e_i, e_j)

    def merge_networks(self, net_i: Any, net_j: Any,
                       networks: Dict[str, Any], round_idx: int) -> Dict[str, Any]:
        """Merge two networks into one with averaged domain representation.

        Keeps ``net_i``'s id (and therefore its LoRA adapter); net_j's
        knowledge bookkeeping is folded in.
        """
        merged = self._clone_network(net_i, net_i.id, "merged")
        # merge knowledge bookkeeping
        for cid, lvl in net_j.internalization_level.items():
            merged.internalization_level[cid] = max(
                merged.internalization_level.get(cid, 0.0), lvl
            )
            merged.knowledge_states[cid] = 3
        for cid in net_j.accepted_ids:
            if cid not in merged.accepted_ids:
                merged.accepted_ids.append(cid)
        merged.accepted_embeddings = (
            list(net_i.accepted_embeddings) + list(net_j.accepted_embeddings)
        )
        if net_i.domain_embedding is not None and net_j.domain_embedding is not None:
            merged.domain_embedding = 0.5 * (net_i.domain_embedding + net_j.domain_embedding)

        new_nets = dict(networks)
        del new_nets[net_i.id]
        del new_nets[net_j.id]
        new_nets[merged.id] = merged
        self.evolution_log.append(
            {"op": "merge", "sources": [net_i.id, net_j.id], "into": merged.id,
             "round": round_idx}
        )
        self._last_change_round = round_idx
        return new_nets

    # ------------------------------------------------------------------ #
    # 4.4 dynamic connections
    # ------------------------------------------------------------------ #
    def update_connections(self, networks: Dict[str, Any],
                           co_activation_matrix: Dict[Tuple[str, str], int],
                           info_flow: Optional[Dict[Tuple[str, str], float]] = None,
                           round_idx: int = 0) -> Dict[Tuple[str, str], float]:
        """w_ij = a*CoActivation_ij + b*InfoFlow_ij; connect when w > tau."""
        info_flow = info_flow or {}
        connections: Dict[Tuple[str, str], float] = {}
        ids = list(networks.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                co = float(co_activation_matrix.get((a, b), 0) +
                           co_activation_matrix.get((b, a), 0))
                flow = float(info_flow.get((a, b), 0.0) + info_flow.get((b, a), 0.0))
                w = self.co_act_w * co + self.info_flow_w * flow
                if w >= self.connect_threshold:
                    connections[(a, b)] = w
        if connections:
            self.evolution_log.append(
                {"op": "connect", "pairs": list(connections.keys()), "round": round_idx}
            )
        return connections

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.evolution_log:
            counts[entry["op"]] = counts.get(entry["op"], 0) + 1
        return counts
