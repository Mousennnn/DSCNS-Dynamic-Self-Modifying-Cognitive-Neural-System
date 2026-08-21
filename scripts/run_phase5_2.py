"""Phase 5.2: Outcome-Conditioned Error-Driven Self-Modification.

Core experiment: failure injection → correction → recovery cycle.

Experiments:
  A. Ablation: no_mod, p5_m, p5_mm, p5_mme, p5_2
  B. Failure injection: force failures at fixed rounds, observe correction
  E. Correction comparison: learned vs rollback vs -Δθ
  Multi-seed (5 seeds)

Run: python scripts/run_phase5_2.py [--rounds 150] [--seeds 5] [--experiments A,B,E]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import subprocess
from typing import List, Dict, Any, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import make_base_model, make_config, prepare_data
from dscns.utils import set_seed
from phase5_common import (build_phase5_networks, make_phase5_stream,
                           make_random_delta as random_delta)
from run_p5_long_horizon import probe_eval, theta_norm, param_hash


def safety_envelope(magnitude, delta_norm, param_norm, max_ratio):
    if delta_norm < 1e-12:
        return magnitude
    return min(magnitude, max_ratio * param_norm / max(delta_norm, 1e-12))
from dscns.error_correction import ErrorState, ErrorEncoder
from dscns.correction_generator import CorrectionGenerator
from dscns.modification_outcome import OutcomeEvaluator, ModificationOutcome, FailureInjector
from dscns.modification_memory import EpisodicSelfModificationMemory, EpisodicModificationRecord

BASE_DIR = os.path.join("experiments", "phase5_2")
SEED_BASE = 42
PROBE_SEED = 1234
BATCH_SIZE = 8
MAX_LEN = 192
TARGET_NAMES = ["attn_lora_A", "attn_lora_B", "mlp_lora_B"]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_p52_experiment(seed: int, rounds: int, experiment: str, config: dict):
    """Run one P5.2 experimental condition for a given seed."""
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    p5_cfg.num_networks = 1
    p5_cfg.seed = seed

    base = make_base_model(p5_cfg, tag=f"p52_{experiment}_s{seed}")
    networks = build_phase5_networks(base, p5_cfg)
    net = networks[0]
    tokenizer = base.tokenizer

    # enable P5.1 module
    from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
    net.plasticity = IntrinsicPlasticityModule(
        hidden_dim=768, adapter_dim=16, meta_dim=32, plasticity_rank=8,
        p51=True, m_min=config["p51_m_min"], m_max=config["p51_m_max"],
        m_init_bias=-3.0, error_dim=config["p51_error_dim"],
        num_target_groups=3).to(base.device)

    # correction generator
    corrector = CorrectionGenerator(
        input_dim=config["p51_error_dim"] + 1 + 1 + 1 + 3 + 16,
        correction_dim=128).to(base.device)
    corrector_optimizer = torch.optim.Adam(corrector.parameters(), lr=config["correction_lr"])

    # evaluator + failure injector
    evaluator = OutcomeEvaluator()
    failure_injector = None
    if experiment in ("failure_injection", "p5_2", "correction_compare", "B"):
        inj_rounds = config.get("failure_injection_rounds", list(range(10, 150, 20)))
        print(f"  [setup] injection_rounds={inj_rounds[:5]}... ({len(inj_rounds)} total)", flush=True)
        failure_injector = FailureInjector(
            inj_rounds,
            injection_magnitude=config.get("failure_injection_magnitude", 1.0),
            injection_alpha=config.get("failure_injection_alpha", 0.1))

    # memory + error tracking
    from dscns.modification_memory import EpisodicSelfModificationMemory, EpisodicModificationRecord
    memory = EpisodicSelfModificationMemory(capacity=config["p51_memory_capacity"],
                                            top_k=config["p51_memory_top_k"])
    prev_error = None
    prev_proposal = None
    pending_correction = None
    failure_rounds = []
    correction_rounds = []
    recovery_rounds = []
    gross = 0.0
    theta0 = net.snapshot_parameters()
    hash0 = param_hash(net)

    data = prepare_data(p5_cfg)
    probes = prepare_probes(data)
    p5_cfg.num_rounds = rounds
    stream = make_phase5_stream(p5_cfg, data, np.random.RandomState(seed))
    pm0, logits0, _ = probe_eval(net, base, probes, MAX_LEN)
    loss_0 = pm0.get("probe_loss", 0.0)
    prev_loss = loss_0              # performance = -loss (higher = better)
    prev_entropy = pm0.get("probe_entropy", 4.0)

    for r, batch in enumerate(stream):
        rnd = r + 1
        row = {"round": rnd}
        texts = [t["text"] if isinstance(t, dict) else str(t) for t in batch]

        # ---- observe ----
        with torch.no_grad():
            out_h = net.generate_delta(texts, tokenizer, max_len=MAX_LEN, grad_enabled=False)

        # ---- probe before ----
        pm_before, _, _ = probe_eval(net, base, probes, MAX_LEN,
                                     ref_batches=logits0, prev_batches=None)
        loss_before = pm_before.get("probe_loss", 0.0)
        entropy_before = pm_before.get("probe_entropy", 4.0)

        # ---- generate proposal ----
        with torch.no_grad():
            proposal = net.plasticity.generate_proposal(
                out_h["components"]["pooled_h"].unsqueeze(1),
                net._current_params_tensors(),
                net._get_meta_info(net.plasticity_cfg.get("meta_dim", 32)),
                error_state=prev_error, memory_z=None, mask=None)

        # ---- injection override ----
        injected = False
        if failure_injector and failure_injector.should_inject(rnd):
            inj = failure_injector.get_injection_params()
            proposal["magnitude"] = inj["magnitude"]
            proposal["alpha_override"] = max(inj["alpha"], 0.5)  # aggressive
            if inj["target_group"] is not None:
                proposal["target_group"] = inj["target_group"]
            injected = True
            if rnd <= 15:
                print(f"    [inject r{rnd}] alpha={inj['alpha']} mag={inj['magnitude']}", flush=True)

        # ---- apply correction from previous round if pending ----
        correction_applied = False
        correction_norm = 0.0
        if pending_correction is not None:
            ca = pending_correction["correction_W_A"]
            cb = pending_correction["correction_W_B"]
            cw = pending_correction["correction_weight"]
            effective_alpha = p5_cfg.plasticity_alpha * cw
            with torch.no_grad():
                for n, p in net.peft_model.named_parameters():
                    if f".{net.id}." in n:
                        if "lora_A" in n and p.size(1) == ca.size(0):
                            p.data.add_(ca.t().to(p.device) * effective_alpha)
                            correction_applied = True
                        elif "lora_B" in n and p.size(0) == cb.size(1):
                            p.data.add_(cb.t().to(p.device) * effective_alpha)
                            correction_applied = True
            if correction_applied:
                correction_norm = float(ca.norm()) + float(cb.norm())
                correction_rounds.append(rnd)
                if prev_error is not None and prev_error.probe_delta < -0.001:
                    recovery_rounds.append(rnd)
            pending_correction = None

        # ---- apply proposal (mandatory) ----
        before_snap = net.snapshot_parameters()
        mag_applied = safety_envelope(proposal["magnitude"],
                                      float(out_h["delta_W_A"].norm()) + float(out_h["delta_W_B"].norm()),
                                      theta_norm(net), 0.1)
        proposal["magnitude"] = mag_applied
        net.apply_self_modification(proposal, alpha=p5_cfg.plasticity_alpha)
        # ---- failure injection: direct weight corruption (stress test) ----
        if injected:
            with torch.no_grad():
                for n, p in net.peft_model.named_parameters():
                    if f".{net.id}." in n and "lora" in n:
                        p.data.add_(torch.randn_like(p.data) * 0.08)
        applied_change = _applied_change(net, before_snap)
        gross += applied_change

        # ---- probe after ----
        pm_after, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
        loss_after = pm_after.get("probe_loss", 0.0)
        entropy_after = pm_after.get("probe_entropy", 4.0)

        # ---- classify outcome: score = -loss (higher = better) ----
        # ds = prev_score - score_after = -prev_loss - (-loss_after) = loss_before - loss_after
        ev = evaluator.evaluate(score_before=-prev_loss, score_after=-loss_after,
                                loss_before=prev_loss, loss_after=loss_after,
                                entropy_before=prev_entropy, entropy_after=entropy_after,
                                param_norm=theta_norm(net),
                                has_nan=torch.isnan(torch.tensor(theta_norm(net))).item(),
                                delta_score=prev_loss - loss_after)
        outcome_class = ev["outcome"]
        category = ev["category"]

        # check recovery
        if prev_error is not None and prev_error.probe_delta < -0.001:
            if outcome_class in ("success", "partial_success") and loss_after < prev_loss:
                category = "recovery"
                if rnd not in recovery_rounds:
                    recovery_rounds.append(rnd)

        if category == "failure" or (injected and outcome_class in ("failure", "catastrophic")):
            failure_rounds.append(rnd)

        # ---- store error for correction ----
        error_state = ErrorState(
            task_delta=0.0,
            probe_delta=prev_loss - loss_after,  # positive = improvement
            logit_delta=0.0,
            entropy_delta=entropy_after - entropy_before,
            parameter_drift=applied_change,
            prev_target=proposal["target_group"],
            prev_magnitude=mag_applied,
        )

        # ---- generate correction for next round if failure ----
        if (category == "failure" or (injected and outcome_class in ("failure", "catastrophic"))):
            with torch.no_grad():
                err_t = error_state.to_tensor().unsqueeze(0).to(base.device)
                z_e = net.plasticity.error_encoder(err_t)
                mem_z = torch.zeros(1, config["p51_error_dim"], device=base.device)
                dA = out_h["delta_W_A"].detach()
                dB = out_h["delta_W_B"].detach()
                pw = torch.tensor([[proposal["magnitude"]]], device=base.device)
                corr = corrector(z_e, dA, dB, pw, proposal["target_group"], mem_z)
                pending_correction = corr

        # ---- store experience ----
        record = EpisodicModificationRecord(
            round_id=rnd,
            core_z=proposal.get("core_z", torch.zeros(256)),
            state_pooled=out_h["components"]["pooled_h"].detach().cpu(),
            meta_info=out_h["meta_info"].detach().cpu(),
            target_group=proposal["target_group"],
            magnitude=mag_applied,
            magnitude_applied=mag_applied,
            delta_norm=applied_change,
            probe_delta=prev_loss - loss_after,
            outcome=outcome_class,
            category=category,
            correction_applied=correction_applied,
            correction_norm=correction_norm,
            error_state=error_state,
            reward=max(0.0, (prev_loss - loss_after) * 100),
        )
        memory.add(record)

        prev_error = error_state
        prev_loss = loss_after
        prev_entropy = entropy_after
        prev_proposal = proposal

        if rnd % 10 == 0 or rnd == rounds:
            marker = " [INJECT]" if injected else ""
            print(f"  [{experiment}] r{rnd}: loss={loss_after:.6f} "
                  f"mag={mag_applied:.3f} outcome={outcome_class} "
                  f"cat={category} inject={injected} corr={correction_applied}"
                  f"{marker}", flush=True)

    # ---- summary ----
    total_failures = len(failure_rounds)
    total_corrections = len(correction_rounds)
    total_recovery = len(recovery_rounds)
    failure_rate = total_failures / max(rounds, 1)
    correction_rate = total_corrections / max(total_failures, 1)
    recovery_rate = total_recovery / max(total_failures, 1)

    # repeat failure rate
    repeat_failures = 0
    if len(failure_rounds) >= 2:
        prev_f_target = None
        for fr in failure_rounds:
            # find the record for this round
            rec = next((r for r in memory.records if r.round_id == fr), None)
            if rec and rec.error_state:
                if prev_f_target is not None and rec.target_group == prev_f_target:
                    repeat_failures += 1
                prev_f_target = rec.target_group

    rfr = repeat_failures / max(total_failures - 1, 1) if total_failures >= 2 else 0.0

    result = {
        "experiment": experiment, "seed": seed, "rounds": rounds,
        "git_commit": git_sha(),
        "hash0": hash0, "hash_final": param_hash(net),
        "param_norm_0": _tn(theta0), "param_norm_final": theta_norm(net),
        "net_drift": _nd(net, theta0), "gross_drift": gross,
        "probe_drift_final": pm_after.get("output_drift_vs_0", 0),
        "failure_rate": failure_rate, "correction_rate": correction_rate,
        "recovery_rate": recovery_rate, "repeat_failure_rate": rfr,
        "total_failures": total_failures, "total_corrections": total_corrections,
        "total_recovery": total_recovery, "repeat_failures": repeat_failures,
        "memory_stats": memory.get_category_counts(),
        "wall_seconds": time.time() - t0_global,
    }
    return result, memory


def prepare_probes(data):
    rng = np.random.RandomState(PROBE_SEED)
    probes = []
    counts = {"general": 8, "math": 6, "logic": 6, "code": 6, "science": 6}
    for domain, k in counts.items():
        pool = data["train"][domain]
        chosen = [str(t) for t in rng.choice(pool, size=min(k, len(pool)), replace=False)]
        for t in chosen:
            probes.append({"id": len(probes) + 1, "domain": domain, "text": t})
    return probes[:32]


def _tn(theta0):
    with torch.no_grad():
        return sum(p.norm().item() for p in theta0["lora_A"].values()) + \
               sum(p.norm().item() for p in theta0["lora_B"].values())


def _nd(net, theta0):
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


t0_global = 0.0


def main():
    global t0_global
    t0_global = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--experiments", default="A,B")
    ap.add_argument("--out", default=BASE_DIR)
    args = ap.parse_args()

    config_path = "config/phase5_2.yaml"
    import yaml
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["num_rounds"] = args.rounds
    config["num_seeds"] = args.seeds

    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "configs"), exist_ok=True)
    with open(os.path.join(args.out, "configs", "phase5_2.yaml"), "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    experiments = [e.strip() for e in args.experiments.split(",")]
    all_results = {}

    seeds = [SEED_BASE + i for i in range(args.seeds)]

    for exp in experiments:
        print(f"\n=== Experiment {exp} ({args.rounds} rounds × {args.seeds} seeds) ===")
        exp_results = []
        for si, seed in enumerate(seeds):
            print(f"  seed {seed} ({si+1}/{len(seeds)}) ...")
            result, memory = run_p52_experiment(seed, args.rounds, exp, config)
            exp_results.append(result)
            exp_dir = os.path.join(args.out, "results", f"{exp}_s{seed}")
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "result.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            with open(os.path.join(exp_dir, "memory.json"), "w") as f:
                json.dump(memory.snapshot(), f, indent=2, default=str)

        # aggregate across seeds
        agg = {}
        for key in ["failure_rate", "correction_rate", "recovery_rate",
                     "repeat_failure_rate", "net_drift", "probe_drift_final",
                     "total_failures", "total_corrections", "total_recovery"]:
            vals = [r[key] for r in exp_results]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        agg["seeds"] = len(seeds)
        agg["rounds"] = args.rounds
        agg["wall_seconds_mean"] = float(np.mean([r["wall_seconds"] for r in exp_results]))

        all_results[exp] = agg
        with open(os.path.join(args.out, "results", f"{exp}_summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

        print(f"\n  {exp}: FR={agg['failure_rate_mean']:.3f}±{agg['failure_rate_std']:.3f} "
              f"CR={agg['correction_rate_mean']:.3f} RR={agg['recovery_rate_mean']:.3f} "
              f"RFR={agg['repeat_failure_rate_mean']:.3f} "
              f"drift={agg['net_drift_mean']:.2f}±{agg['net_drift_std']:.2f}")

    with open(os.path.join(args.out, "results", "all_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== P5.2 Summary ===")
    for exp, agg in all_results.items():
        print(f"{exp}: FR={agg['failure_rate_mean']:.3f} "
              f"CR={agg['correction_rate_mean']:.3f} "
              f"RR={agg['recovery_rate_mean']:.3f} "
              f"RFR={agg['repeat_failure_rate_mean']:.3f} "
              f"drift={agg['net_drift_mean']:.2f}±{agg['net_drift_std']:.2f}")


if __name__ == "__main__":
    main()
