"""Phase 3 experiment: dynamic structure evolution (report section 7.2 Phase 3).

Comparison on a shifted distribution stream:
- fixed:  5 networks, no split/merge/connect (control)
- evolve: 5 networks + structure evolution enabled

Metrics: multi-task performance, specialization, parameter efficiency,
adaptation speed after distribution shift.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (DOMAINS, build_system, eval_per_domain_loss, make_base_model,
                    make_config, prepare_data, save_results)
from dscns.evaluation import structural_metrics
from dscns.utils import set_seed
import numpy as np


def make_shifted_stream(config, data, rng):
    """general(3) -> code(3) -> mixed code-heavy(4) -> science(4): shift stream."""
    train = data["train"]
    per_round = config.samples_per_round
    stream = []
    phases = [("general", 3), ("code", 3), ("mixed_code", 4), ("science", 4)]
    for phase, n in phases:
        for _ in range(n):
            if phase == "mixed_code":
                samples = []
                for domain, frac in [("code", 0.5), ("general", 0.25),
                                     ("science", 0.25)]:
                    pool = train[domain]
                    k = max(1, int(per_round * frac))
                    samples += [{"text": t, "domain": domain, "source": domain,
                                 "reliability": 0.8}
                                for t in rng.choice(pool, size=min(k, len(pool)),
                                                    replace=False)]
                rng.shuffle(samples)
                stream.append(samples[:per_round])
            else:
                pool = train[phase]
                k = min(per_round, len(pool))
                stream.append([{"text": t, "domain": phase, "source": phase,
                                "reliability": 0.8}
                               for t in rng.choice(pool, size=k, replace=False)])
    return stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = make_config()
    config.samples_per_round = 32
    config.num_networks = 5
    config.seed = args.seed
    set_seed(config.seed)
    data = prepare_data(config)
    eval_sets = data["eval"]
    rng = np.random.RandomState(config.seed)
    stream = make_shifted_stream(config, data, rng)

    all_results = {}
    for tag, evolution in (("fixed", False), ("evolve", True)):
        t0 = time.time()
        config.evolution_enabled = evolution
        config.evolution_every = 4
        base = make_base_model(config, tag=f"p3_{tag}")
        system = build_system(config, base, data)
        matrix, spec_curve, nnet_curve = [], [], []
        for r, batch in enumerate(stream):
            info = system.process_experiences(batch)
            perf = eval_per_domain_loss(system, eval_sets, None, 48)
            matrix.append([perf[d] for d in DOMAINS])
            system.meta_update({"overall": float(np.mean(list(perf.values())))})
            sm = structural_metrics(system.networks, system.domain_embeddings)
            spec_curve.append(sm["mean_specialization"])
            nnet_curve.append(sm["n_networks"])
            print(f"[{tag}] round {r + 1}: " +
                  ", ".join(f"{d}={perf[d]:.4f}" for d in DOMAINS) +
                  f" | nets={sm['n_networks']:.0f} spec={sm['mean_specialization']:.3f}",
                  flush=True)
        results = {
            "tag": tag,
            "evolution_enabled": evolution,
            "performance_matrix": matrix,
            "final_performance": {d: float(matrix[-1][j])
                                  for j, d in enumerate(DOMAINS)},
            "specialization_curve": spec_curve,
            "n_networks_curve": nnet_curve,
            "structure": structural_metrics(system.networks,
                                            system.domain_embeddings),
            "evolution_log": getattr(getattr(system, "evolver", None),
                                     "evolution_log", []),
            "wall_seconds": time.time() - t0,
        }
        # capture evolution log if evolver logged
        save_results(args.out, tag, results)
        all_results[tag] = results

    # adaptation speed: code perf at round 4 (start of code phase) vs round 7
    for tag, res in all_results.items():
        pm = res["performance_matrix"]
        res["code_adapt_speed"] = {
            "round4": pm[3][DOMAINS.index("code")],
            "round7": pm[6][DOMAINS.index("code")],
            "delta": pm[6][DOMAINS.index("code")] - pm[3][DOMAINS.index("code")],
        }
    save_results(args.out, "summary", all_results)
    print("=== Phase 3 summary ===")
    for tag, res in all_results.items():
        print(f"{tag}: final=" + ", ".join(f"{d}={v:.4f}" for d, v in
                                           res["final_performance"].items()))
        print(f"  code_adapt_speed: {res['code_adapt_speed']}")


if __name__ == "__main__":
    main()
