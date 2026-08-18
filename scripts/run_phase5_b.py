"""Phase 5-B experiment: intrinsic parameter self-modification.

Core proposition (design report P5):

    theta_t -> h_t -> delta_theta_t -> theta_{t+1} -> h_{t+1} -> ...

Two arms on a 20-round shifted stream (general(5) -> code(5) ->
mixed_code(5) -> science(5)):

  * fixed -- task learning only, no plasticity (control baseline);
  * p5b   -- task learning + intrinsic plasticity at a fixed external
             trigger, with the accept/rollback safety protocol.

Primary evidence is the closed loop itself (delta existence, state
dependency, parameter transition, behavioral change, stability, and the
negative controls in run_negative_controls.py); performance is recorded
as a secondary, descriptive outcome (P5 explicitly does not require
performance gains).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import (make_base_model, make_config, prepare_data,
                    save_results)
from dscns.evaluation import compute_continual_learning_metrics
from dscns.utils import set_seed
from phase5_common import (DOMAINS, apply_plasticity_step, build_phase5_networks,
                           eval_matched_per_domain, make_phase5_stream,
                           train_round_step, update_meta_from_probe)


def _param_norm(net) -> float:
    import torch

    tensors = net._current_params_tensors()
    return float(tensors["W_A"].norm()) + float(tensors["W_B"].norm())


def run_arm(tag: str, delta_mode: str, config, data, eval_sets, stream,
            constant_delta=None, other_texts_fn=None):
    t0 = time.time()
    base = make_base_model(config, tag=f"p5_{tag}")
    config.enable_plasticity = delta_mode is not None
    networks = build_phase5_networks(base, config)
    probe = data["probe"]
    matrix, param_norm_curve, mod_log = [], [], []
    accepted = applied = 0

    for r, batch in enumerate(stream):
        texts = [e["text"] for e in batch]
        primary_domain = batch[0]["domain"]
        for net in networks:
            train_round_step(net, base, texts, config)
            if (config.enable_plasticity and net.step_count > 0 and
                    net.step_count % config.plasticity_interval == 0):
                other = None
                if other_texts_fn is not None:
                    other = other_texts_fn(batch, r, net, data)
                rec = apply_plasticity_step(net, base, batch, config,
                                            mode=delta_mode,
                                            constant_delta=constant_delta,
                                            other_texts=other)
                rec.update({"round": r + 1, "net_id": net.id,
                            "step": net.step_count})
                mod_log.append(rec)
                applied += 1
                if rec.get("accepted"):
                    accepted += 1
        perf = eval_matched_per_domain(networks, eval_sets, base, 48)
        matrix.append([perf[d] for d in DOMAINS])
        param_norm_curve.append(float(np.mean([_param_norm(n) for n in networks])))
        update_meta_from_probe(networks, probe, base, primary_domain)
        acc = (accepted / applied) if applied else 0.0
        print(f"[{tag}] round {r + 1}: " +
              ", ".join(f"{d}={perf[d]:.4f}" for d in DOMAINS) +
              f" | triggers={applied} accept={acc:.2f}"
              f" | param_norm={param_norm_curve[-1]:.3f}", flush=True)

    results = {
        "tag": tag,
        "delta_mode": delta_mode,
        "performance_matrix": matrix,
        "final_performance": {d: float(matrix[-1][j])
                              for j, d in enumerate(DOMAINS)},
        "param_norm_curve": param_norm_curve,
        "plasticity_log": mod_log,
        "triggers": applied,
        "accepted": accepted,
        "acceptance_rate": (accepted / applied) if applied else 0.0,
        "wall_seconds": time.time() - t0,
        "config": {
            "plasticity_interval": config.plasticity_interval,
            "plasticity_alpha": config.plasticity_alpha,
            "plasticity_rank": config.plasticity_rank,
            "meta_dim": config.meta_dim,
            "modulation_strength_init": config.modulation_strength_init,
            "train_plasticity": config.train_plasticity,
            "use_hidden": config.use_hidden,
            "use_param_stats": config.use_param_stats,
            "use_meta": config.use_meta,
        },
    }
    results.update(compute_continual_learning_metrics(matrix, domains=DOMAINS))

    # closed-loop summary statistics (the P5 core evidence)
    if mod_log:
        d_norms = [m["delta_norm"] for m in mod_log]
        strengths = [m["modulation_strength"] for m in mod_log]
        results["closed_loop"] = {
            "num_events": len(mod_log),
            "delta_norm_mean": float(np.mean(d_norms)),
            "delta_norm_std": float(np.std(d_norms)),
            "delta_norm_min": float(np.min(d_norms)),
            "delta_norm_max": float(np.max(d_norms)),
            "delta_norm_variance": float(np.var(d_norms)),
            "modulation_strength_curve": strengths,
            "pred_change_mean": float(np.mean(
                [m.get("pred_change_rate", 0.0) for m in mod_log])),
            "logits_diff_mean": float(np.mean(
                [m.get("logits_diff", 0.0) for m in mod_log])),
            "loss_before_mean": float(np.mean([m["loss_before"] for m in mod_log])),
            "loss_after_mean": float(np.mean([m["loss_after"] for m in mod_log])),
            "perplexity_mean": float(np.mean([m["perplexity"] for m in mod_log])),
            "reasons": {str(m.get("reason")): sum(1 for x in mod_log
                                                  if x.get("reason") == m.get("reason"))
                        for m in mod_log},
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="fixed,p5b")
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()

    config = make_config(cfg_path="config/phase5.yaml")
    config.seed = args.seed
    config.num_rounds = args.rounds
    set_seed(config.seed)
    data = prepare_data(config)
    eval_sets = data["eval"]
    rng = np.random.RandomState(config.seed)
    stream = make_phase5_stream(config, data, rng)

    all_results = {}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for tag, mode in (("fixed", None), ("p5b", "intrinsic")):
        if tag not in arms:
            continue
        all_results[tag] = run_arm(tag, mode, config, data, eval_sets, stream)
        save_results(args.out, tag, all_results[tag])
    save_results(args.out, "summary", all_results)

    print("=== Phase 5-B summary ===")
    for tag, res in all_results.items():
        print(f"{tag}: final=" + ", ".join(f"{d}={v:.4f}" for d, v in
                                           res["final_performance"].items()))
        print(f"  AF={res['AF']:.4f} FWT={res['FWT']:.4f} CLS={res['CLS']:.4f}")
        if "closed_loop" in res:
            cl = res["closed_loop"]
            print(f"  closed loop: {cl['num_events']} events, "
                  f"delta_norm {cl['delta_norm_mean']:.5f}+-{cl['delta_norm_std']:.5f}, "
                  f"accept={res['acceptance_rate']:.2f}, "
                  f"pred_change={cl['pred_change_mean']:.4f}")


if __name__ == "__main__":
    main()
