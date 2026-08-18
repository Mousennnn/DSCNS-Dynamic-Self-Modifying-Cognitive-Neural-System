"""Evaluation metrics for continual learning (report section 7.4 / 10.3).

Computes the standard continual-learning metrics from a performance matrix
``perf[round][domain]``:
- Average Forgetting (AF)
- Forward Transfer (FWT)
- Continual Learning Score (CLS)
- per-domain forgetting / retention / acquisition
plus DSCNS-specific structural metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_continual_learning_metrics(performance_matrix: List[List[float]],
                                       random_baseline: Optional[List[float]] = None,
                                       domains: Optional[List[str]] = None) -> Dict[str, float]:
    """Faithful implementation of report section 10.3."""
    T = len(performance_matrix)  # number of rounds
    if T == 0:
        return {}
    domains = domains or [f"task_{j}" for j in range(len(performance_matrix[0]))]
    n_domains = len(performance_matrix[0])

    # 1. average forgetting (per report: peak after task j minus final)
    forgetting = []
    for j in range(n_domains):
        col = [performance_matrix[i][j] for i in range(T)]
        peak = max(col)
        final = col[-1]
        forgetting.append(peak - final)
    AF = float(np.mean(forgetting))

    # 2. forward transfer vs random/initial baseline
    base = random_baseline or [performance_matrix[0][j] for j in range(n_domains)]
    fwt_terms = []
    for j in range(1, min(T, n_domains)):
        fwt_terms.append(performance_matrix[j][j] - base[j])
    FWT = float(np.mean(fwt_terms)) if fwt_terms else 0.0

    # 3. continual learning score
    CLS = float(np.mean(performance_matrix[T - 1])) - AF

    return {"AF": AF, "FWT": FWT, "CLS": CLS}


def forgetting_rate(perf: List[float]) -> float:
    """(initial - final) / initial per domain."""
    if len(perf) < 2 or perf[0] == 0:
        return 0.0
    return (perf[0] - perf[-1]) / perf[0]


def per_domain_metrics(performance_matrix: List[List[float]],
                       domains: List[str]) -> Dict[str, Dict[str, float]]:
    T = len(performance_matrix)
    out = {}
    for j, d in enumerate(domains):
        col = [performance_matrix[i][j] for i in range(T)]
        peak = max(col)
        out[d] = {
            "initial": col[0],
            "peak": peak,
            "final": col[-1],
            "forgetting_rate": forgetting_rate(col),
            "peak_forgetting": (peak - col[-1]) / peak if peak > 0 else 0.0,
            "acquisition": col[-1] - col[0],  # improvement while learning it
        }
    return out


def acquisition_and_retention(performance_matrix: List[List[float]],
                              domains: List[str],
                              phase_rounds: List[int]) -> Dict[str, float]:
    """New-knowledge acquisition and old-knowledge retention.

    ``phase_rounds`` = [5, 5, 5, 5, 5]: how many rounds each domain is the
    active learning target.
    """
    T = len(performance_matrix)
    acquired, retained = [], []
    start = 0
    for j, phase_len in enumerate(phase_rounds):
        end = min(start + phase_len, T)
        if j < len(domains) and end > start:
            # acquisition: perf on domain j at the end of its own phase
            acquired.append(performance_matrix[end - 1][j])
            # retention: perf on domain j at the very end
            retained.append(performance_matrix[T - 1][j])
        start = end
    return {
        "mean_acquisition": float(np.mean(acquired)) if acquired else 0.0,
        "mean_retention": float(np.mean(retained)) if retained else 0.0,
        "per_domain_acquired": acquired,
        "per_domain_retained": retained,
    }


def structural_metrics(networks: Dict[str, Any],
                       domain_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """DSCNS structural metrics (report section 7.4)."""
    ids = list(networks.keys())
    n = len(ids)
    complementarity, redundancy = 0.0, 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = networks[ids[i]], networks[ids[j]]
            ra, rb = a.representation_embedding(), b.representation_embedding()
            if ra.size > 1 and rb.size > 1:
                from .utils import cosine_similarity

                sim = cosine_similarity(ra, rb)
                redundancy += sim
                complementarity += (1.0 - sim)
                pairs += 1
    specialization = 0.0
    for net in networks.values():
        if len(net.recent_domains) > 10:
            from .utils import entropy

            counts: Dict[str, int] = {}
            for d in net.recent_domains:
                counts[d] = counts.get(d, 0) + 1
            probs = [c / len(net.recent_domains) for c in counts.values()]
            spec = 1.0 - entropy(probs) / np.log(max(len(counts), 2))
            specialization += spec
    specialization /= max(n, 1)
    return {
        "n_networks": float(n),
        "mean_complementarity": complementarity / max(pairs, 1),
        "mean_redundancy": redundancy / max(pairs, 1),
        "mean_specialization": specialization,
    }


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize an experiment result dict into report-ready numbers."""
    pm = results["performance_matrix"]
    domains = results.get("domains", [])
    phase_rounds = results.get("phase_rounds", [])
    base = results.get("baseline", None)
    out = compute_continual_learning_metrics(pm, random_baseline=base, domains=domains)
    out["per_domain"] = per_domain_metrics(pm, domains)
    out.update(acquisition_and_retention(pm, domains, phase_rounds))
    if results.get("structure"):
        out["structure"] = results["structure"]
    return out
