"""Phase 5-C experiment: adaptive plasticity learning (report section 11).

P5-C studies whether P_phi itself improves from experience:

    phi_t -> phi_{t+1}

After each *accepted* intrinsic modification the short-horizon reward
(perf after a short adaptation minus perf before) is measured; positive-
reward cases are stored, and once enough cases accumulate the plasticity
module is trained offline (reward-weighted MSE) to reproduce successful
deltas on replayed states with the *current* parameter statistics.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import make_base_model, make_config, prepare_data, save_results
from dscns.evaluation import compute_continual_learning_metrics
from dscns.plasticity_trainer import PlasticityTrainer
from dscns.utils import set_seed
from phase5_common import (DOMAINS, build_phase5_networks,
                           eval_matched_per_domain, make_phase5_stream,
                           quick_validation, train_round_step,
                           update_meta_from_probe)


def plasticity_step_with_learning(net, base, batch, config, trainer,
                                  rng=None):
    """P5-C plasticity step: apply -> adapt -> reward -> record / rollback."""
    texts = [e["text"] for e in batch]
    params_before = net.snapshot_parameters()
    rec = {"mode": "intrinsic_p5c", "applied": False, "accepted": False}

    with net._no_grad_ctx():
        delta = net.generate_delta(texts, base.tokenizer, max_len=config.max_len)
    delta_norm = float(delta["delta_W_A"].norm()) + float(delta["delta_W_B"].norm())
    rec["delta_norm"] = delta_norm
    if delta_norm < getattr(config, "min_delta_threshold", 1e-6):
        rec["reason"] = "delta_too_small"
        return rec

    sub = texts[: getattr(config, "quick_validation_samples", 8)]
    loss_before = float(np.mean(net.losses_for_texts(
        sub, base.tokenizer, batch_size=8, max_len=config.max_len)))
    perf_before = float(np.exp(-loss_before))

    net.apply_intrinsic_modification(delta, alpha=config.plasticity_alpha)
    rec["applied"] = True

    ok, reason, loss_after, ppl = quick_validation(
        net, base, sub, config, before_loss=loss_before)
    if not ok:
        net.restore_parameters(params_before)
        rec.update({"accepted": False, "reason": reason,
                    "loss_before": loss_before, "loss_after": loss_after,
                    "perplexity": ppl, "reward": 0.0})
        return rec

    # short adaptation (experiment-controlled) then reward.
    # NOTE: adaptation must not perturb the external trigger cadence, so the
    # step counter is preserved across these auxiliary learning steps.
    saved_step = net.step_count
    for _ in range(getattr(config, "adaptation_steps", 3)):
        train_round_step(net, base, texts[:8], config)
    net.step_count = saved_step
    loss_after = float(np.mean(net.losses_for_texts(
        sub, base.tokenizer, batch_size=8, max_len=config.max_len)))
    perf_after = float(np.exp(-loss_after))
    reward = perf_after - perf_before

    rec.update({"accepted": True, "reason": "ok",
                "loss_before": loss_before, "loss_after": loss_after,
                "perplexity": float(np.exp(loss_after)),
                "reward": reward})

    trainer.record_success_case(
        texts=sub,
        delta_params=delta,
        reward=reward,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = make_config(cfg_path="config/phase5.yaml")
    config.seed = args.seed
    config.train_plasticity = True
    set_seed(config.seed)
    data = prepare_data(config)
    eval_sets = data["eval"]
    rng = np.random.RandomState(config.seed)
    stream = make_phase5_stream(config, data, rng)

    t0 = time.time()
    base = make_base_model(config, tag="p5_p5c")
    config.enable_plasticity = True
    networks = build_phase5_networks(base, config)
    probe = data["probe"]
    trainers = {net.id: PlasticityTrainer(net, config, base=base)
                for net in networks}
    matrix, param_norm_curve, mod_log = [], [], []
    accepted = applied = 0

    for r, batch in enumerate(stream):
        texts = [e["text"] for e in batch]
        primary_domain = batch[0]["domain"]
        for net in networks:
            train_round_step(net, base, texts, config)
            if net.step_count > 0 and \
                    net.step_count % config.plasticity_interval == 0:
                rec = plasticity_step_with_learning(
                    net, base, batch, config, trainers[net.id], rng=rng)
                rec.update({"round": r + 1, "net_id": net.id,
                            "step": net.step_count})
                mod_log.append(rec)
                applied += 1
                if rec.get("accepted"):
                    accepted += 1
        # offline plasticity learning once enough success cases exist
        for net in networks:
            if len(trainers[net.id].success_memory) >= \
                    getattr(config, "plasticity_train_threshold", 10):
                trainer = trainers[net.id]
                loss = trainer.train_from_memory()
                if loss is not None:
                    print(f"[p5c] round {r + 1} net {net.id}: "
                          f"plasticity train loss={loss:.6f} "
                          f"(memory={len(trainer.success_memory)})", flush=True)
        perf = eval_matched_per_domain(networks, eval_sets, base, 48)
        matrix.append([perf[d] for d in DOMAINS])
        param_norm_curve.append(float(np.mean([
            float(n._current_params_tensors()["W_A"].norm()) +
            float(n._current_params_tensors()["W_B"].norm())
            for n in networks])))
        update_meta_from_probe(networks, probe, base, primary_domain)
        acc = (accepted / applied) if applied else 0.0
        print(f"[p5c] round {r + 1}: " +
              ", ".join(f"{d}={perf[d]:.4f}" for d in DOMAINS) +
              f" | triggers={applied} accept={acc:.2f}", flush=True)

    results = {
        "tag": "p5c",
        "delta_mode": "intrinsic_p5c",
        "performance_matrix": matrix,
        "final_performance": {d: float(matrix[-1][j])
                              for j, d in enumerate(DOMAINS)},
        "param_norm_curve": param_norm_curve,
        "plasticity_log": mod_log,
        "triggers": applied,
        "accepted": accepted,
        "acceptance_rate": (accepted / applied) if applied else 0.0,
        "wall_seconds": time.time() - t0,
        "plasticity_trainer": {
            net_id: trainer.statistics() for net_id, trainer in trainers.items()
        },
        "rewards": [m.get("reward", 0.0) for m in mod_log if m.get("accepted")],
        "mean_reward": float(np.mean([m.get("reward", 0.0) for m in mod_log
                                      if m.get("accepted")])) if mod_log else 0.0,
    }
    results.update(compute_continual_learning_metrics(matrix, domains=DOMAINS))
    if mod_log:
        d_norms = [m["delta_norm"] for m in mod_log]
        results["closed_loop"] = {
            "num_events": len(mod_log),
            "delta_norm_mean": float(np.mean(d_norms)),
            "delta_norm_variance": float(np.var(d_norms)),
            "pred_change_mean": float(np.mean(
                [m.get("pred_change_rate", 0.0) for m in mod_log])),
            "loss_before_mean": float(np.mean([m["loss_before"] for m in mod_log])),
            "loss_after_mean": float(np.mean([m["loss_after"] for m in mod_log])),
        }
    save_results(args.out, "p5c", results)
    save_results(args.out, "p5c_summary", {"p5c": results})

    print("=== Phase 5-C summary ===")
    print(f"p5c: final=" + ", ".join(f"{d}={v:.4f}" for d, v in
                                     results["final_performance"].items()))
    print(f"  AF={results['AF']:.4f} FWT={results['FWT']:.4f} "
          f"CLS={results['CLS']:.4f} | accept={results['acceptance_rate']:.2f} "
          f"mean_reward={results['mean_reward']:.5f}")
    for net_id, st in results["plasticity_trainer"].items():
        print(f"  net {net_id}: memory={st['memory_size']} "
              f"train_calls={st['train_calls']} "
              f"loss_curve={[round(x, 5) for x in st['train_loss_curve']]}")


if __name__ == "__main__":
    main()
