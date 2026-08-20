"""P5.1 experiment: Mandatory self-modification + self-determined magnitude
+ error-conditioned self-correction.

Implementation of the P5.1 design report (report sections 1-52).

Arms (ablation matrix, section 48):
  p5_m    — mandatory modification, fixed magnitude (alpha), no error learning
  p5_mm   — mandatory + model-selected magnitude + target, no error learning
  p5_mme  — mandatory + magnitude + error learning (full P5.1)
  random  — budget-matched random directions

All arms share: frozen P5 mechanism (commit 3208463), frozen probe set,
identical θ₀ (seed 42), the same operational loop structure, and the
same safety envelope (no NaN/Inf/explosion allowed).

Run: python scripts/run_phase5_1.py [--smoke] [--arms p5_m,p5_mm,p5_mme,random]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import make_base_model, make_config, prepare_data
from dscns.utils import set_seed
from phase5_common import build_phase5_networks
from run_p5_long_horizon import (
    make_probe_set, make_operational_stream, probe_eval,
    theta_norm, param_hash, layer_wise, random_delta)

BASE_DIR = os.path.join("experiments", "phase5_1")
SEED = 42
PROBE_SEED = 1234
BATCH_SIZE = 8
MAX_LEN = 192
TARGET_NAMES = ["attn_lora_A", "attn_lora_B", "mlp_lora_B"]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.getcwd(),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_config(path="config/phase5_1.yaml"):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_error(prev_error: dict, perf_before: float, perf_after: float,
                  probe_before: float, probe_after: float,
                  probe_loss_before: float, probe_loss_after: float,
                  entropy_before: float, entropy_after: float,
                  delta_norm: float, prev_target: int, prev_magnitude: float):
    from dscns.error_correction import ErrorState
    return ErrorState(
        task_delta=perf_after - perf_before,
        probe_delta=probe_after - probe_before,
        forgetting_delta=0.0,
        logit_delta=0.0,
        entropy_delta=entropy_after - entropy_before,
        parameter_drift=delta_norm,
        prev_target=prev_target,
        prev_magnitude=prev_magnitude,
    )


def check_catastrophic(net, entropy: float) -> bool:
    cfg = None
    return (entropy < 0.1 or
            theta_norm(net) > 1000.0 or
            torch.isnan(torch.tensor(theta_norm(net))).item())


def safety_envelope(requested_magnitude: float, delta_norm: float,
                    param_norm: float, max_ratio: float) -> float:
    if delta_norm < 1e-12:
        return requested_magnitude
    hard_max = max_ratio * param_norm / delta_norm
    return min(requested_magnitude, hard_max)


def train_plasticity_online(module, memory, config):
    """Error-conditioned training of Pφ from the episodic memory."""
    import random as _random
    from dscns.error_correction import ErrorState
    if module is None or not getattr(module.plasticity, "p51", False):
        return 0.0
    if len(memory.records) < 4:
        return 0.0
    # balance: sample equally from success and failure
    successes = [r for r in memory.records if r.category == "success"]
    failures = [r for r in memory.records if r.category in ("failure", "recovery")]
    n_s = min(len(successes), len(failures), config["p51_plasticity_batch"] // 2)
    if n_s < 1:
        n_s = min(len(memory.records), config["p51_plasticity_batch"])
        batch = _random.sample(memory.records, n_s)
    else:
        batch = _random.sample(successes, n_s) + _random.sample(failures, n_s)
    import torch.nn.functional as F
    module.plasticity.train()
    module.plasticity_optimizer = getattr(module, "plasticity_optimizer", None)
    if module.plasticity_optimizer is None:
        module.plasticity_optimizer = torch.optim.Adam(
            module.plasticity.parameters(), lr=config.get("p51_plasticity_lr", 3e-4))
    opt = module.plasticity_optimizer
    total_loss = 0.0
    dev = next(module.plasticity.parameters()).device
    for rec in batch:
        if rec.core_z is None or rec.error_state is None:
            continue
        # Use the RECORDED core_z (from generation-time context) + error encoding
        core_z = rec.core_z.to(dev).unsqueeze(0)          # (1, 256)
        err_t = rec.error_state.to_tensor().to(dev).unsqueeze(0)  # (1, 8)
        z_e = module.plasticity.error_encoder(err_t)        # (1, 32)
        z_mem = torch.zeros(1, module.plasticity.error_dim, device=dev)
        extended_z = torch.cat([core_z, z_e, z_mem], dim=-1)  # (1, 320)
        target_pred = module.plasticity.target_head(extended_z)  # (1, 3)
        target_loss = F.cross_entropy(
            target_pred,
            torch.tensor([rec.target_group], device=dev, dtype=torch.long))
        mag_pred = module.plasticity.magnitude_head(extended_z)
        mag_loss = F.mse_loss(
            mag_pred.squeeze(-1),
            torch.tensor([rec.magnitude_applied], device=dev, dtype=torch.float32))
        loss = (target_loss * config.get("p51_lambda_target", 0.5) +
                mag_loss * config.get("p51_lambda_mag", 0.5))
        loss.backward()
        total_loss += float(loss.item())
    opt.step()
    opt.zero_grad()
    module.plasticity.eval()
    return total_loss / max(len(batch), 1)


def run_arm_5_1(arm_name: str, config_5_1: dict, p5_cfg, data, probes,
                rounds: int, probe_every: int, ckpt_every: int, smoke: bool):
    """Run one P5.1 experimental arm (150 or 3000 rounds)."""
    import random

    t0 = time.time()
    set_seed(SEED)
    base = make_base_model(p5_cfg, tag=f"p51_{arm_name}")
    p5_cfg.num_networks = 1
    networks = build_phase5_networks(base, p5_cfg)
    net = networks[0]
    tokenizer = base.tokenizer

    mandatory = "p5_" in arm_name and "fixed" not in arm_name and arm_name != "no_mod"
    self_mag = "mm" in arm_name or "mme" in arm_name
    error_learning = "mme" in arm_name
    is_random = arm_name == "random"
    is_no_mod = arm_name == "no_mod"

    # enable P5.1 heads if magnitude or error learning is needed
    if mandatory and (self_mag or error_learning) and net.plasticity is not None:
        if not getattr(net.plasticity, "p51", False):
            # re-create module with p51=True
            from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
            net.plasticity = IntrinsicPlasticityModule(
                hidden_dim=768, adapter_dim=16,
                meta_dim=32,
                plasticity_rank=8,
                use_hidden=True, use_param_stats=True, use_meta=True,
                modulation_strength_init=0.05,
                p51=True,
                m_min=config_5_1["p51_m_min"],
                m_max=config_5_1["p51_m_max"],
                m_init_bias=config_5_1["p51_m_init_bias"],
                error_dim=config_5_1["p51_error_dim"],
                num_target_groups=config_5_1["p51_num_target_groups"],
            ).to(base.device)

    # episodic memory
    from dscns.modification_memory import (EpisodicSelfModificationMemory,
                                           EpisodicModificationRecord)
    memory = EpisodicSelfModificationMemory(
        capacity=config_5_1["p51_memory_capacity"],
        top_k=config_5_1["p51_memory_top_k"])

    evaluator = None
    if error_learning:
        from dscns.error_correction import OutcomeEvaluator
        evaluator = OutcomeEvaluator(
            success_thresh=config_5_1["p51_success_threshold"],
            failure_thresh=config_5_1["p51_failure_threshold"])

    stream = make_operational_stream(data, rounds)
    theta0 = net.snapshot_parameters()
    hash0 = param_hash(net)
    out_dir = os.path.join(BASE_DIR, "results", arm_name)
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", arm_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # reference probe eval
    probe_meta, logits0, prev_snap = probe_eval(net, base, probes, MAX_LEN)
    prev_error_state = None
    prev_target = -1
    prev_magnitude = 0.0
    gross = 0.0
    rows = []
    success_count = failure_count = recovery_count = 0
    repeated_errors = 0
    prev_fail_target = -1

    for r, texts in enumerate(stream):
        round_no = r + 1
        row = {"round": round_no}

        # ---- observe ----
        with torch.no_grad():
            out_h = net.generate_delta(texts, tokenizer, max_len=MAX_LEN,
                                       grad_enabled=False)
        delta_norm = float(out_h["delta_W_A"].norm()) + float(out_h["delta_W_B"].norm())

        # ---- probe before ----
        pm_before, _, _ = probe_eval(net, base, probes, MAX_LEN,
                                     ref_batches=logits0, prev_batches=prev_snap)
        probe_before = pm_before.get("output_drift_vs_0", 0.0)
        probe_loss_before = pm_before.get("probe_loss", 0.0)
        probe_entropy_before = pm_before.get("probe_entropy", 4.0)

        # ---- generate proposal ----
        if is_no_mod:
            # no modification at all
            proposal = {"delta_W_A": torch.zeros(768, 16, device=net.peft_model.device),
                        "delta_W_B": torch.zeros(16, 768, device=net.peft_model.device),
                        "magnitude": 0.0, "target_group": -1,
                        "confidence": 0.0,
                        "modulation_strength": 0.0}
        elif mandatory and (self_mag or error_learning):
            # P5.1 path: generate_proposal with magnitude + target heads
            error_state = prev_error_state
            memory_z = None
            if error_learning and memory.records:
                last_rec = memory.records[-1]
                if last_rec.core_z is not None:
                    memory_z = last_rec.core_z
            with torch.no_grad():
                proposal = net.plasticity.generate_proposal(
                    out_h["components"]["pooled_h"].unsqueeze(1),
                    net._current_params_tensors(),
                    net._get_meta_info(net.plasticity_cfg.get("meta_dim", 32)),
                    error_state=error_state, memory_z=memory_z,
                    mask=None)
        elif mandatory:
            # P5-style mandatory (p5_m): P5 forward, no magnitude/target heads
            proposal = {"delta_W_A": out_h["delta_W_A"], "delta_W_B": out_h["delta_W_B"],
                        "magnitude": 1.0, "target_group": -1,
                        "confidence": 0.0,
                        "modulation_strength": out_h["modulation_strength"]}
        elif is_random:
            proposal = make_random_proposal(delta_norm, net.peft_model.device)
        else:
            # P5 fixed baseline (no mandatory, P5 forward)
            proposal = {"delta_W_A": out_h["delta_W_A"], "delta_W_B": out_h["delta_W_B"],
                        "magnitude": 1.0, "target_group": -1,
                        "confidence": 0.0,
                        "modulation_strength": out_h["modulation_strength"]}

        # ---- safety envelope ----
        mag_requested = proposal["magnitude"]
        mag_applied = safety_envelope(mag_requested, delta_norm, theta_norm(net),
                                      config_5_1["p51_max_param_ratio"])
        proposal["magnitude"] = mag_applied

        # ---- apply ----
        before_snap = net.snapshot_parameters()
        if mandatory and (self_mag or error_learning):
            net.apply_self_modification(proposal, alpha=p5_cfg.plasticity_alpha)
        elif not is_no_mod:
            net.apply_intrinsic_modification(
                {"delta_W_A": proposal["delta_W_A"], "delta_W_B": proposal["delta_W_B"],
                 "modulation_strength": proposal["modulation_strength"]},
                alpha=p5_cfg.plasticity_alpha)
        applied_change = _applied_change(net, before_snap)
        gross += applied_change
        dnet = _net_drift(net, theta0)

        # ---- probe after ----
        pm_after, _, prev_snap = probe_eval(net, base, probes, MAX_LEN,
                                            ref_batches=logits0,
                                            prev_batches=prev_snap)
        probe_after = pm_after.get("output_drift_vs_0", 0.0)
        probe_loss_after = pm_after.get("probe_loss", 0.0)
        probe_entropy_after = pm_after.get("probe_entropy", 4.0)

        # ---- error / outcome ----
        error = compute_error(
            prev_error_state, 0.0, 0.0,
            probe_before, probe_after,
            probe_loss_before, probe_loss_after,
            probe_entropy_before, probe_entropy_after,
            applied_change, prev_target, mag_applied)

        outcome = "neutral"
        category = "success"
        rolled_back = False
        if evaluator:
            ev = evaluator.evaluate(0.0, 0.0, probe_before, probe_after,
                                    probe_entropy_after, theta_norm(net),
                                    check_catastrophic(net, probe_entropy_after))
            outcome = ev["outcome"]
            if ev["catastrophic"]:
                net.restore_parameters(before_snap)
                rolled_back = True
                category = "catastrophic"
            elif ev["failure_c"]:
                category = "failure"
                if prev_fail_target == proposal["target_group"]:
                    repeated_errors += 1
                prev_fail_target = proposal["target_group"]
            elif ev["success_b"]:
                category = "success"
            elif prev_error_state is not None and prev_error_state.probe_delta < -0.01:
                category = "recovery"
        success_count += category == "success"
        failure_count += category == "failure"
        recovery_count += category == "recovery"

        # ---- store experience ----
        exp_record = EpisodicModificationRecord(
            round_id=round_no,
            core_z=proposal.get("core_z", torch.zeros(256)),
            state_pooled=out_h["components"]["pooled_h"].detach().cpu(),
            meta_info=out_h["meta_info"].detach().cpu(),
            target_group=proposal["target_group"],
            target_probs=proposal.get("target_probs", torch.zeros(3)).detach().cpu(),
            magnitude=mag_requested,
            magnitude_applied=mag_applied,
            confidence=proposal.get("confidence", 0),
            delta_norm=applied_change,
            probe_delta=probe_after - probe_before,
            probe_loss_delta=probe_loss_after - probe_loss_before,
            entropy_delta=probe_entropy_after - probe_entropy_before,
            outcome=outcome,
            category=category,
            rolled_back=rolled_back,
            error_state=error,
            reward=max(0.0, (probe_after - probe_before) * 100),
        )
        memory.add(exp_record)
        prev_error_state = error
        prev_target = proposal["target_group"]
        prev_magnitude = mag_applied

        # ---- train Pφ (error-conditioned) ----
        if error_learning and round_no % config_5_1.get("p51_train_every", 1) == 0:
            train_loss = train_plasticity_online(net, memory, config_5_1)
        else:
            train_loss = 0.0

        row.update({
            "delta_norm": delta_norm,
            "theta_norm": theta_norm(net),
            "applied_change": applied_change,
            "gross_drift": gross,
            "net_drift": dnet,
            "drift_ratio": dnet / (max(gross, 1e-12)),
            "magnitude_requested": mag_requested,
            "magnitude_applied": mag_applied,
            "target_group": proposal["target_group"],
            "confidence": proposal.get("confidence", 0),
            "probe_drift": probe_after,
            "probe_loss": probe_loss_after,
            "probe_entropy": probe_entropy_after,
            "probe_loss_delta": probe_loss_after - probe_loss_before,
            "probe_entropy_delta": probe_entropy_after - probe_entropy_before,
            "outcome": outcome,
            "category": category,
            "rolled_back": rolled_back,
            "train_loss": train_loss,
            "nan_count": int(torch.isnan(
                torch.cat([proposal["delta_W_A"].flatten(),
                            proposal["delta_W_B"].flatten()])).sum()),
            "inf_count": int(torch.isinf(
                torch.cat([proposal["delta_W_A"].flatten(),
                            proposal["delta_W_B"].flatten()])).sum()),
        })
        rows.append(row)

        if round_no % 50 == 0 or round_no == rounds:
            acc_rate = success_count / max(1, success_count + failure_count)
            print(f"[{arm_name}] r{round_no}: d_norm={delta_norm:.3f} "
                  f"theta={theta_norm(net):.3f} mag={mag_applied:.3f} "
                  f"target={TARGET_NAMES[proposal['target_group']]} "
                  f"probe={probe_after:.3f} outcome={outcome} cat={category} "
                  f"success={success_count} fail={failure_count} "
                  f"recovery={recovery_count} repeated_err={repeated_errors}", flush=True)

    # final
    pmf, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
    final_hash = param_hash(net)
    results = {
        "arm": arm_name, "rounds": rounds, "seed": SEED, "git_commit": git_sha(),
        "hash0": hash0, "hash_final": final_hash, "hash_changed": final_hash != hash0,
        "param_norm_0": _tn(theta0), "param_norm_final": theta_norm(net),
        "gross_drift": gross, "net_drift": _net_drift(net, theta0),
        "drift_ratio": _net_drift(net, theta0) / max(gross, 1e-12),
        "probe_drift_final": pmf.get("output_drift_vs_0", 0),
        "probe_loss_final": pmf.get("probe_loss", 0),
        "probe_entropy_final": pmf.get("probe_entropy", 0),
        "success_count": success_count, "failure_count": failure_count,
        "recovery_count": recovery_count, "repeated_errors": repeated_errors,
        "failure_rate": failure_count / max(1, failure_count + success_count),
        "repeated_error_rate": repeated_errors / max(1, failure_count, 1),
        "nan_total": sum(r["nan_count"] for r in rows),
        "inf_total": sum(r["inf_count"] for r in rows),
        "magnitude_stats": {
            "requested_mean": float(np.mean([r["magnitude_requested"] for r in rows])),
            "applied_mean": float(np.mean([r["magnitude_applied"] for r in rows])),
            "applied_std": float(np.std([r["magnitude_applied"] for r in rows])),
        },
        "target_distribution": {t: sum(1 for r in rows if r["target_group"] == i)
                                for i, t in enumerate(TARGET_NAMES)},
        "memory_stats": memory.get_category_counts(),
        "wall_seconds": time.time() - t0,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(out_dir, "group.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "memory.json"), "w") as f:
        json.dump(memory.snapshot(), f, indent=2, default=str)

    print(f"[{arm_name}] done: net_drift={results['net_drift']:.4f} "
          f"probe_drift={results['probe_drift_final']:.4f} "
          f"success={success_count} fail={failure_count} "
          f"recovery={recovery_count} repeated_err={repeated_errors} "
          f"({time.time()-t0:.0f}s)")
    return results


def _tn(theta0):
    t = theta0
    return float(t["lora_A"][list(t["lora_A"])[0]].norm()) * 0 + \
        float(sum(p.norm() for p in t["lora_A"].values()) +
               sum(p.norm() for p in t["lora_B"].values()))


def _net_drift(net, theta0):
    d = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in theta0.get("lora_A", {}):
                d += float((p.data - theta0["lora_A"][n]).norm())
            elif n in theta0.get("lora_B", {}):
                d += float((p.data - theta0["lora_B"][n]).norm())
    return d


def _applied_change(net, before):
    c = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in before.get("lora_A", {}):
                c += float((p.data - before["lora_A"][n]).norm())
            elif n in before.get("lora_B", {}):
                c += float((p.data - before["lora_B"][n]).norm())
    return c


def make_random_proposal(delta_norm, device):
    eA = torch.randn(768, 16, device=device)
    eB = torch.randn(16, 768, device=device)
    n = eA.norm() + eB.norm()
    s = float(delta_norm) / float(max(n, 1e-12))
    target = np.random.randint(0, 3)
    return {
        "delta_W_A": eA * s, "delta_W_B": eB * s,
        "magnitude": np.random.uniform(0.02, 1.0),
        "target_group": target,
        "target_probs": torch.nn.functional.one_hot(
            torch.tensor(target), 3).float(),
        "confidence": 1.0 / 3,
        "modulation_strength": 0.05,
    }


def main():
    global BASE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="p5_m,p5_mm,p5_mme,random")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--out", default=BASE_DIR)
    args = ap.parse_args()

    BASE_DIR = args.out
    os.makedirs(BASE_DIR, exist_ok=True)
    for d in ["config", "probe_set", "results", "checkpoints", "plots"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

    cfg_5_1 = load_config()
    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    p5_cfg.num_networks = 1
    p5_cfg.seed = SEED
    set_seed(SEED)
    data = prepare_data(p5_cfg)
    probes = make_probe_set(data)

    rounds = args.rounds or (6 if args.smoke else 150)
    probe_every = 1
    ckpt_every = 50 if rounds > 50 else 1

    arms = [a.strip() for a in args.arms.split(",")]
    all_results = {}
    for arm in arms:
        all_results[arm] = run_arm_5_1(arm, cfg_5_1, p5_cfg, data, probes,
                                       rounds, probe_every, ckpt_every, args.smoke)
        save_results(BASE_DIR, arm, all_results[arm])
    save_results(BASE_DIR, "summary", all_results)

    print("\n=== P5.1 Summary ===")
    for arm, res in all_results.items():
        print(f"{arm}: net_drift={res['net_drift']:.4f} probe={res['probe_drift_final']:.4f} "
              f"success={res['success_count']} fail={res['failure_count']} "
              f"recovery={res['recovery_count']} rep_err={res['repeated_error_rate']:.3f} "
              f"mag_mean={res['magnitude_stats']['applied_mean']:.3f}")


def save_results(base, name, results):
    with open(os.path.join(base, "results", f"{name}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
