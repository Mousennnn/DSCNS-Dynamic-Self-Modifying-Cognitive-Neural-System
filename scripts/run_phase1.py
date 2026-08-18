"""Phase 1 experiment runner (report section 7.2 / 10.1).

Experiments:
- Control: standard sequential fine-tuning (no gating, no verification).
- Exp1:    single network + selective internalization (evaluation gating).
- Exp2:    multi-network (5) + cross-network verification + meta control.

Metrics: performance matrix, AF / FWT / CLS, forgetting, acquisition/retention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (DOMAINS, build_exemplars, build_experience_stream,
                    build_system, control_train_step, eval_per_domain_loss,
                    generation_eval, make_base_model, make_config,
                    make_control_network, plot_curves, plot_performance_matrix,
                    prepare_data, save_results)
from dscns.evaluation import (acquisition_and_retention,
                              compute_continual_learning_metrics,
                              per_domain_metrics, structural_metrics)
from dscns.utils import set_seed


def run_mode(mode: str, config, data, out_dir: str) -> dict:
    t0 = time.time()
    base = make_base_model(config, tag=mode)
    stream = build_experience_stream(data, config)
    eval_sets = data["eval"]
    domains = DOMAINS
    n_rounds = len(stream)
    matrix: list = []
    per_network_trace: dict = {}
    log_lines = []

    def log(msg: str):
        line = f"[{mode}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    if mode == "control":
        net = make_control_network(base)
        log(f"Control: sequential fine-tuning, {config.train_steps_per_round} steps/round")
        for r, batch in enumerate(stream):
            texts = [e["text"] for e in batch]
            for _ in range(config.train_steps_per_round):
                control_train_step(net, texts, base, lr=config.lora_lr)
            perf = eval_per_domain_loss(net, eval_sets, base, config.eval_per_domain)
            matrix.append([perf[d] for d in domains])
            net.performance_history.append(float(np.mean(list(perf.values()))))
            log(f"round {r + 1}/{n_rounds}: " +
                ", ".join(f"{d}={perf[d]:.4f}" for d in domains))
        gen = {}
        if config.with_generation_eval:
            gen = generation_eval(net, eval_sets, base)
            log(f"generation accuracy: {gen}")
        structure = {}
        params = sum(p.numel() for p in net._adapter_params())
    else:
        n_nets = 1 if mode == "exp1" else (5 if mode == "exp2" else config.num_networks)
        config.num_networks = n_nets
        system = build_system(config, base, data)
        log(f"{mode}: {n_nets} network(s), selective internalization"
            + (" + cross-verification" if mode == "exp2" else ""))
        for r, batch in enumerate(stream):
            info = system.process_experiences(batch)
            perf = system.best_domain_performance(eval_sets, config.eval_per_domain)
            matrix.append([perf[d] for d in domains])
            system.meta_update({"overall": float(np.mean(list(perf.values())))})
            dec = {}
            for k, v in info["decisions"].items():
                dec[v["action"]] = dec.get(v["action"], 0) + 1
            log(f"round {r + 1}/{n_rounds}: " +
                ", ".join(f"{d}={perf[d]:.4f}" for d in domains) +
                f" | decisions={dec} | internalize={info['internalization']}")
            for net_id, net in system.networks.items():
                per_network_trace.setdefault(net_id, []).append(
                    net.performance_history[-1] if net.performance_history else 0.0)
        gen = {}
        if config.with_generation_eval:
            gen = generation_eval(system, eval_sets, base)
            log(f"generation accuracy: {gen}")
        structure = structural_metrics(system.networks, system.domain_embeddings)
        structure["bus_messages"] = system.bus.get_message_stats()
        structure["verification"] = system.verifier.stats()
        structure["trust_weights"] = {k: round(v, 3)
                                      for k, v in system.verifier.trust_weights.items()}
        params = sum(p.numel() for p in base.peft_model.parameters()
                     if p.requires_grad) or None

    final_perf = matrix[-1]
    results = {
        "mode": mode,
        "config": config.to_dict(),
        "domains": domains,
        "phase_rounds": config.phase_rounds,
        "performance_matrix": matrix,
        "final_performance": {d: float(v) for d, v in zip(domains, final_perf)},
        "generation_accuracy": gen,
        "structure": structure,
        "metrics": compute_continual_learning_metrics(
            matrix, random_baseline=matrix[0], domains=domains),
        "per_domain": per_domain_metrics(matrix, domains),
        "acquisition_retention": acquisition_and_retention(
            matrix, domains, config.phase_rounds),
        "per_network_trace": per_network_trace,
        "log": log_lines,
        "wall_seconds": time.time() - t0,
    }
    path = save_results(out_dir, mode, results)
    plot_performance_matrix(matrix, domains, os.path.join(out_dir, f"{mode}_matrix.png"),
                            f"{mode} performance matrix")
    plot_curves(results, out_dir, domains)
    log(f"done in {time.time() - t0:.0f}s -> {path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["control", "exp1", "exp2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="experiments/phase1")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override total rounds (spread evenly over phases)")
    ap.add_argument("--with-gen", action="store_true", default=True)
    ap.add_argument("--no-gen", dest="with_gen", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = make_config(args.config)
    if args.rounds:
        n_phases = len(config.phase_rounds)
        per = max(1, args.rounds // n_phases)
        config.phase_rounds = [per] * n_phases
    config.with_generation_eval = args.with_gen
    config.seed = args.seed
    set_seed(config.seed)

    os.makedirs(args.out, exist_ok=True)
    data = prepare_data(config)
    print("data sizes:", {d: len(t) for d, t in data["train"].items()}, flush=True)

    all_results = {}
    for mode in args.modes:
        all_results[mode] = run_mode(mode, config, data, args.out)
    summary = {m: {"metrics": r["metrics"], "gen": r["generation_accuracy"],
                   "wall_seconds": r["wall_seconds"]}
               for m, r in all_results.items()}
    save_results(args.out, "summary", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
