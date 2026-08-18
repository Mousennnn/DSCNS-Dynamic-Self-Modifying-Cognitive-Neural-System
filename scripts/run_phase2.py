"""Phase 2 experiment: active learning (report section 7.2 Phase 2 / 3.6).

Strategies:
- baseline: random sampling
- exp1:     uncertainty sampling (highest base-model loss)
- exp2:     information-gain sampling (uncertainty x novelty vs memory)
- exp3:     meta-cognitive guided sampling (weak-domain focus + IG)

Each strategy selects ``samples_per_round`` experiences per round from the
full training pool and processes them through the Exp1-style system.
Metric: learning efficiency curve (per-round eval performance), knowledge
coverage, and sample efficiency (rounds to reach a coverage target).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (DOMAINS, build_system, eval_per_domain_loss, make_base_model,
                    make_config, prepare_data, save_results)
from dscns.utils import set_seed, top_k_indices
import numpy as np


def select(strategy: str, pool: list, budget: int, learner, tokenizer=None,
           memory=None, meta_state=None, pool_embs=None, rng=None) -> list:
    """Select experiences according to the active-learning strategy.

    Uncertainty / information-gain are computed with the LEARNER's current
    adapter state (not the frozen base), so scores adapt to learning progress.
    """
    from dscns.metacognition import MetaCognitiveController

    rng = rng or np.random.RandomState(0)
    if strategy == "random":
        idx = rng.choice(len(pool), size=budget, replace=False)
        return [pool[int(i)] for i in idx]

    texts = [e["text"] for e in pool]
    losses = learner.losses_for_texts(texts, tokenizer, batch_size=8)

    if strategy == "uncertainty":
        idx = top_k_indices(losses, budget)
        return [pool[i] for i in idx]

    if strategy == "info_gain":
        unc = np.clip(1.0 - np.exp(-losses), 0.0, 1.0)
        mem_embs = [e["embedding"] for e in
                    (memory.episodic.episodes if memory and memory.episodic else [])]
        if mem_embs and pool_embs is not None:
            from dscns.utils import cosine_matrix

            sims = cosine_matrix(pool_embs, np.stack(mem_embs)).max(axis=1)
            novelty = 1.0 - sims
        else:
            novelty = np.ones(len(texts))
        scores = unc * novelty
        idx = top_k_indices(scores, budget)
        return [pool[i] for i in idx]

    if strategy == "meta":
        controller = MetaCognitiveController()
        return controller.select_experiences(pool, budget, None, meta_state,
                                             learner=learner, losses=losses)
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "uncertainty", "info_gain", "meta"])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--pool-per-domain", type=int, default=60)
    ap.add_argument("--out", default="experiments/phase2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = make_config()
    config.samples_per_round = 32
    config.num_networks = 1
    config.seed = args.seed
    set_seed(config.seed)
    data = prepare_data(config)

    # full mixed pool
    rng = np.random.RandomState(config.seed)
    pool = []
    for d in DOMAINS:
        texts = data["train"][d]
        for t in rng.choice(texts, size=min(args.pool_per_domain, len(texts)),
                            replace=False):
            pool.append({"text": t, "domain": d, "source": d, "reliability": 0.8})
    rng.shuffle(pool)
    eval_sets = data["eval"]

    all_results = {}
    for strategy in args.strategies:
        t0 = time.time()
        base = make_base_model(config, tag=f"p2_{strategy}")
        system = build_system(config, base, data)
        learner = system.networks["N1"]
        # static pool embeddings from the frozen base (one-time cost)
        pool_embs = base.embed([e["text"] for e in pool], batch_size=8)
        coverage_curve, perf_curve = [], []
        for r in range(1, args.rounds + 1):
            chosen = select(strategy, pool, config.samples_per_round, learner,
                            base.tokenizer, system.memory, system.meta_state,
                            pool_embs, rng)
            info = system.process_experiences(chosen)
            perf = eval_per_domain_loss(system, eval_sets, None, 48)
            mean_perf = float(np.mean(list(perf.values())))
            system.meta_update({"overall": mean_perf})
            perf_curve.append(mean_perf)
            coverage = system.meta_state.knowledge_coverage
            coverage_curve.append(float(np.mean(list(coverage.values()))))
            dec = {}
            for v in info["decisions"].values():
                dec[v["action"]] = dec.get(v["action"], 0) + 1
            print(f"[{strategy}] round {r}: perf={mean_perf:.4f} "
                  f"coverage={coverage_curve[-1]:.3f} decisions={dec}", flush=True)
        results = {
            "strategy": strategy,
            "perf_curve": perf_curve,
            "coverage_curve": coverage_curve,
            "final_perf": perf_curve[-1],
            "final_coverage": coverage_curve[-1],
            "rounds_to_50pct_coverage": next(
                (i + 1 for i, c in enumerate(coverage_curve) if c >= 0.5),
                args.rounds),
            "wall_seconds": time.time() - t0,
        }
        save_results(args.out, strategy, results)
        all_results[strategy] = results
    save_results(args.out, "summary", all_results)
    print("=== Phase 2 summary ===")
    for s, res in all_results.items():
        print(f"{s}: final_perf={res['final_perf']:.4f} "
              f"final_coverage={res['final_coverage']:.3f} "
              f"rounds_to_50%={res['rounds_to_50pct_coverage']}")


if __name__ == "__main__":
    main()
