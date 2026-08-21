"""v0.5.1: Memory-Conditioned Outcome Learning & Error-Driven Self-Modification.

Core experiment: verify whether memory truly changes future modification behavior.

Experiments:
  Memory Ablation (§21):
    A1: Full (C5 memory_conditioned)
    A2: No Memory (C4, memory zeroed)
    A3: Shuffled Memory (irrelevant episodes)
    A4: Random Memory (random embeddings)
    A5: Zero Memory (zero embeddings)

  Correction Ablation (§23):
    C0: No correction
    C2: Pure reversal (-Δθ)
    C3: Learned (no memory)
    C4: Error-conditioned (no memory)
    C5: Error + Memory (full)

  Natural Failure (§19):
    NF: Natural failure (no injection)

  Controlled Failure + Memory (§20):
    CF: Controlled failure injection + full correction

  Weight Ablation (§22):
    W1: Fixed weight
    W3: P5.1 learned weight
    W5: Memory + error-conditioned weight

Run: python scripts/run_v051.py [--rounds 150] [--seeds 5] [--experiments A1,A2,NF]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import make_base_model, make_config, prepare_data
from dscns.utils import set_seed
from phase5_common import build_phase5_networks, make_phase5_stream
from run_p5_long_horizon import probe_eval, theta_norm, param_hash

# v0.5.1 modules
from dscns.error_correction import ErrorState, ErrorEncoder
from dscns.correction_policy import (
    CorrectionPolicy, CorrectionPolicyWithMemory,
    MODE_NONE, MODE_REVERSAL, MODE_LEARNED, MODE_ERROR_COND, MODE_MEMORY_COND,
)
from dscns.modification_outcome import (
    OutcomeEvaluator, ModificationOutcome, FailureInjector,
    V051OutcomeEvaluator, RecoveryMetrics, NaturalFailureDetector,
)
from dscns.modification_memory import (
    EpisodicSelfModificationMemory, EpisodicModificationRecord,
)
from dscns.memory_encoder import (
    MemoryEncoder, ModificationEpisode, MultiSimilarityRetriever,
    MemoryPolicyEncoder,
)
from dscns.experience_replay import (
    ExperienceReplayBuffer, ReplayEntry, train_correction_offline,
)

BASE_DIR = os.path.join("experiments", "phase5_1_v051")
SEED_BASE = 42
PROBE_SEED = 1234
BATCH_SIZE = 8
MAX_LEN = 192
TARGET_NAMES = ["attn_lora_A", "attn_lora_B", "mlp_lora_B"]


# ====================================================================== #
# Probe Set Generation (S=32, M=256)                                     #
# ====================================================================== #

def make_probe_set(data, size: int = 32, seed: int = PROBE_SEED) -> list:
    """Frozen probe set with deterministic seed. size=S/M/L."""
    rng = np.random.RandomState(seed)
    probes = []
    counts_s = {"general": 8, "math": 6, "logic": 6, "code": 6, "science": 6}
    if size <= 32:
        counts = counts_s
    elif size <= 256:
        counts = {d: k * 8 for d, k in counts_s.items()}
    else:
        counts = {d: k * 32 for d, k in counts_s.items()}

    for domain, k in counts.items():
        pool = data["train"].get(domain, [])
        if not pool:
            continue
        chosen = [str(t) for t in rng.choice(pool, size=min(k, len(pool)), replace=False)]
        for t in chosen:
            probes.append({"id": len(probes) + 1, "domain": domain, "text": t})
    return probes[:size]


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #

def git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def safety_envelope(magnitude, delta_norm, param_norm, max_ratio):
    if delta_norm < 1e-12:
        return magnitude
    return min(magnitude, max_ratio * param_norm / max(delta_norm, 1e-12))


def _applied_change(net, before) -> float:
    c = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in before.get("lora_A", {}):
                c += float((p.data - before["lora_A"][n]).norm())
            elif n in before.get("lora_B", {}):
                c += float((p.data - before["lora_B"][n]).norm())
    return c


def _nd(net, theta0) -> float:
    d = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in theta0.get("lora_A", {}):
                d += float((p.data - theta0["lora_A"][n]).norm())
            elif n in theta0.get("lora_B", {}):
                d += float((p.data - theta0["lora_B"][n]).norm())
    return d


# ====================================================================== #
# Core Experiment Runner                                                  #
# ====================================================================== #

def run_v051_experiment(
    seed: int,
    rounds: int,
    experiment: str,
    config: dict,
    data: dict,
    correction_mode: str = "memory_conditioned",
    memory_mode: str = "full",   # full/shuffled/random/zero
    weight_mode: str = "learned",  # fixed/random/learned/error_cond/memory_cond
    use_injection: bool = False,
    probe_size: int = 32,
) -> Tuple[Dict[str, Any], Any]:
    """Run one v0.5.1 experimental condition for a given seed.

    Returns (result_dict, memory_object).
    """
    t0 = time.time()
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- setup model ----
    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    p5_cfg.num_networks = 1
    p5_cfg.seed = seed

    base = make_base_model(p5_cfg, tag=f"v051_{experiment}_s{seed}")
    networks = build_phase5_networks(base, p5_cfg)
    net = networks[0]
    tokenizer = base.tokenizer

    # ---- enable P5.1 module ----
    from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
    net.plasticity = IntrinsicPlasticityModule(
        hidden_dim=768, adapter_dim=16, meta_dim=32, plasticity_rank=8,
        p51=True, m_min=config.get("p51_m_min", 0.02),
        m_max=config.get("p51_m_max", 1.0),
        m_init_bias=-3.0, error_dim=config.get("p51_error_dim", 32),
        num_target_groups=3).to(base.device)

    # ---- correction policy ----
    corrector = CorrectionPolicyWithMemory(
        error_dim=8,
        memory_dim=config.get("correction_memory_dim", 32),
        core_dim=256,
        hidden_dim=config.get("correction_hidden_dim", 64),
        memory_top_k=config.get("memory_top_k", 8),
        lambda_context=config.get("memory_lambda_context", 0.3),
        lambda_proposal=config.get("memory_lambda_proposal", 0.3),
        lambda_error=config.get("memory_lambda_error", 0.2),
        lambda_target=config.get("memory_lambda_target", 0.2),
    ).to(base.device)
    corrector_optimizer = torch.optim.Adam(
        corrector.parameters(), lr=config.get("correction_lr", 3e-4))

    # ---- memory + evaluator ----
    memory = EpisodicSelfModificationMemory(
        capacity=config.get("p51_memory_capacity", 2000),
        top_k=config.get("memory_top_k", 8))
    evaluator = V051OutcomeEvaluator(
        recovery_threshold=config.get("recovery_threshold", 0.0001))
    original_evaluator = OutcomeEvaluator()
    replay_buffer = ExperienceReplayBuffer(
        capacity=config.get("replay_buffer_size", 1000))
    natural_detector = NaturalFailureDetector(
        failure_threshold=config.get("natural_failure_threshold", -0.0001))

    # ---- failure injector ----
    failure_injector = None
    if use_injection:
        inj_rounds = config.get("failure_injection_rounds",
                                list(range(10, rounds, 20)))
        failure_injector = FailureInjector(
            inj_rounds,
            injection_magnitude=config.get("failure_injection_magnitude", 1.0),
            injection_alpha=config.get("failure_injection_alpha", 0.1))

    # ---- probe set ----
    probes = make_probe_set(data, size=probe_size)

    # ---- operational stream ----
    p5_cfg.num_rounds = rounds
    stream = make_phase5_stream(p5_cfg, data, np.random.RandomState(seed))

    # ---- initial state ----
    theta0 = net.snapshot_parameters()
    hash0 = param_hash(net)
    pm0, logits0, _ = probe_eval(net, base, probes, MAX_LEN)
    loss_0 = pm0.get("probe_loss", 0.0)
    prev_loss = loss_0
    prev_entropy = pm0.get("probe_entropy", 4.0)

    # ---- tracking ----
    prev_error = None
    prev_proposal = None
    pending_correction = None
    pending_correction_mode = None
    gross = 0.0
    failure_rounds = []
    correction_rounds = []
    recovery_rounds = []
    round_logs = []
    weight_history = []
    outcome_events = []

    for r, batch in enumerate(stream):
        rnd = r + 1
        texts = [t["text"] if isinstance(t, dict) else str(t) for t in batch]

        # ---- observe: get hidden states + delta ----
        with torch.no_grad():
            out_h = net.generate_delta(texts, tokenizer, max_len=MAX_LEN,
                                       grad_enabled=False)

        # ---- probe before ----
        pm_before, _, _ = probe_eval(net, base, probes, MAX_LEN,
                                     ref_batches=logits0, prev_batches=None)
        loss_before = pm_before.get("probe_loss", 0.0)
        entropy_before = pm_before.get("probe_entropy", 4.0)
        score_before = -loss_before

        # ---- generate proposal ----
        with torch.no_grad():
            proposal = net.plasticity.generate_proposal(
                out_h["components"]["pooled_h"].unsqueeze(1),
                net._current_params_tensors(),
                net._get_meta_info(net.plasticity_cfg.get("meta_dim", 32)),
                error_state=prev_error, memory_z=None, mask=None)

        # ---- apply correction from previous round if pending ----
        correction_applied = False
        correction_norm_val = 0.0
        score_after_correction = score_before

        if pending_correction is not None:
            ca = pending_correction["correction_W_A"]
            cb = pending_correction["correction_W_B"]
            cw = pending_correction.get("correction_strength", 0.5)

            if pending_correction_mode in (MODE_REVERSAL,):
                # C2: pure reversal handled by controller
                pass
            elif pending_correction_mode in (MODE_NONE,):
                pass
            else:
                # C3/C4/C5: apply learned correction
                effective_alpha = p5_cfg.plasticity_alpha * max(cw, 0.1)
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
                    correction_norm_val = float(ca.norm()) + float(cb.norm())
                    correction_rounds.append(rnd)

            # re-evaluate after correction
            if correction_applied:
                pm_corr, _, _ = probe_eval(net, base, probes, MAX_LEN,
                                           ref_batches=logits0)
                score_after_correction = -pm_corr.get("probe_loss", 0.0)

            pending_correction = None
            pending_correction_mode = None

        # ---- injection override ----
        injected = False
        if failure_injector and failure_injector.should_inject(rnd):
            inj = failure_injector.get_injection_params()
            proposal["magnitude"] = inj["magnitude"]
            proposal["alpha_override"] = max(inj["alpha"], 0.5)
            injected = True

        # ---- weight mode ----
        mag = proposal["magnitude"]
        if weight_mode == "fixed":
            mag = 0.05  # fixed weight
        elif weight_mode == "random":
            mag = float(np.random.uniform(0.02, 0.5))
        # "learned", "error_cond", "memory_cond" use the model's own magnitude

        # ---- apply proposal (mandatory, non-zero) ----
        before_snap = net.snapshot_parameters()
        mag_applied = safety_envelope(
            mag, float(out_h["delta_W_A"].norm()) + float(out_h["delta_W_B"].norm()),
            theta_norm(net), 0.1)
        proposal["magnitude"] = mag_applied
        net.apply_self_modification(proposal, alpha=p5_cfg.plasticity_alpha)

        # ---- failure injection: direct weight corruption ----
        if injected:
            with torch.no_grad():
                for n, p in net.peft_model.named_parameters():
                    if f".{net.id}." in n and "lora" in n:
                        p.data.add_(torch.randn_like(p.data) * 0.08)

        applied_change = _applied_change(net, before_snap)
        gross += applied_change

        # ---- probe after ----
        pm_after, _, _ = probe_eval(net, base, probes, MAX_LEN,
                                    ref_batches=logits0)
        loss_after = pm_after.get("probe_loss", 0.0)
        entropy_after = pm_after.get("probe_entropy", 4.0)
        score_after = -loss_after

        # ---- classify outcome ----
        ev = original_evaluator.evaluate(
            score_before=score_before, score_after=score_after,
            loss_before=loss_before, loss_after=loss_after,
            entropy_before=prev_entropy, entropy_after=entropy_after,
            param_norm=theta_norm(net),
            has_nan=torch.isnan(torch.tensor(theta_norm(net))).item(),
            delta_score=loss_before - loss_after)
        outcome_class = ev["outcome"]
        category = ev["category"]

        # ---- check recovery ----
        if prev_error is not None and prev_error.probe_delta < -0.001:
            if outcome_class in ("success", "partial_success") and loss_after < prev_loss:
                category = "recovery"
                if rnd not in recovery_rounds:
                    recovery_rounds.append(rnd)

        if category == "failure" or (injected and outcome_class in ("failure", "catastrophic")):
            failure_rounds.append(rnd)

        # ---- natural failure detection ----
        natural_result = natural_detector.record_round(
            rnd, loss_before - loss_after, injected=injected)

        # ---- weight history ----
        weight_history.append({
            "round": rnd, "weight": mag_applied,
            "category": category, "outcome": outcome_class,
        })

        # ---- outcome event for recovery metrics ----
        outcome_events.append({
            "category": category,
            "correction_applied": correction_applied,
            "score_before_modification": score_before,
            "score_after_modification": score_after,
            "score_after_correction": score_after_correction,
        })

        # ---- build error state ----
        error_state = ErrorState(
            task_delta=0.0,
            probe_delta=loss_before - loss_after,
            logit_delta=0.0,
            entropy_delta=entropy_after - entropy_before,
            parameter_drift=applied_change,
            prev_target=proposal["target_group"],
            prev_magnitude=mag_applied,
        )

        # ---- generate correction for next round if failure ----
        is_failure = (category == "failure" or
                      (injected and outcome_class in ("failure", "catastrophic")))
        if is_failure:
            with torch.no_grad():
                err_t = error_state.to_tensor().unsqueeze(0).to(base.device)
                core_z = proposal.get("core_z", torch.zeros(256)).to(base.device)
                if core_z.dim() == 1:
                    core_z = core_z.unsqueeze(0)

                # determine memory input based on memory_mode
                mem_episodes = memory.records
                if memory_mode == "shuffled":
                    import random as _random
                    mem_episodes = list(mem_episodes)
                    _random.shuffle(mem_episodes)
                elif memory_mode == "random":
                    mem_episodes = []  # will get zero from corrector
                elif memory_mode == "zero":
                    mem_episodes = []  # will get zero from corrector
                # "full": use real memory

                corr_output = corrector(
                    err_t, core_z,
                    out_h["delta_W_A"].detach(),
                    out_h["delta_W_B"].detach(),
                    mag_applied,
                    proposal["target_group"],
                    mem_episodes,
                    mode=correction_mode,
                    prev_error=prev_error,
                )
                pending_correction = corr_output
                pending_correction_mode = correction_mode

            if rnd not in correction_rounds and correction_mode != MODE_NONE:
                correction_rounds.append(rnd)
            if category == "failure" and rnd not in recovery_rounds:
                # mark recovery will be checked next round
                pass

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
            probe_delta=loss_before - loss_after,
            outcome=outcome_class,
            category=category,
            correction_applied=correction_applied,
            correction_norm=correction_norm_val,
            error_state=error_state,
            reward=max(0.0, (loss_before - loss_after) * 100),
        )
        memory.add(record)

        # ---- replay buffer ----
        replay_entry = ReplayEntry(
            error_state=error_state,
            core_z=proposal.get("core_z", torch.zeros(256)).cpu(),
            prev_delta_A=out_h["delta_W_A"].detach().cpu(),
            prev_delta_B=out_h["delta_W_B"].detach().cpu(),
            prev_weight=mag_applied,
            prev_target=proposal["target_group"],
            delta_score=loss_before - loss_after,
            outcome=outcome_class,
            category=category,
            reward=max(0.0, (loss_before - loss_after) * 100),
            round_id=rnd,
        )
        replay_buffer.add(replay_entry)

        # ---- round log ----
        round_logs.append({
            "round": rnd,
            "loss_before": loss_before, "loss_after": loss_after,
            "delta_score": loss_before - loss_after,
            "weight": mag_applied, "target": proposal["target_group"],
            "outcome": outcome_class, "category": category,
            "injected": injected, "correction_applied": correction_applied,
            "correction_norm": correction_norm_val,
            "natural_failure": natural_result["is_natural_failure"],
            "applied_change": applied_change,
            "theta_norm": theta_norm(net),
        })

        prev_error = error_state
        prev_loss = loss_after
        prev_entropy = entropy_after
        prev_proposal = proposal

        if rnd % 10 == 0 or rnd == rounds:
            print(f"  [{experiment}] r{rnd}: loss={loss_after:.6f} "
                  f"mag={mag_applied:.3f} out={outcome_class} cat={category} "
                  f"inject={injected} corr={correction_applied}", flush=True)

    # ---- final metrics ----
    total_failures = len(failure_rounds)
    total_corrections = len(correction_rounds)
    total_recovery = len(recovery_rounds)

    # recovery metrics (v0.5.1 separated)
    recovery_metrics = evaluator.compute_recovery_metrics(outcome_events)

    # RFR variants
    rfr_target = memory.get_rfr_target()
    rfr_similar = memory.get_rfr_similar(
        similarity_threshold=config.get("memory_similarity_threshold", 0.5))
    rfr_exact = memory.get_rfr_exact()

    # weight adaptation
    weight_stats = memory.get_weight_stats_by_outcome()

    # target transitions
    target_transitions = memory.get_target_transition_matrix()

    # experience absorption indicators
    w_after_failure = np.mean([w["weight"] for w in weight_history
                               if w["category"] == "failure"]) if any(
        w["category"] == "failure" for w in weight_history) else 0.0
    w_after_success = np.mean([w["weight"] for w in weight_history
                               if w["category"] == "success"]) if any(
        w["category"] == "success" for w in weight_history) else 0.0

    result = {
        "experiment": experiment,
        "correction_mode": correction_mode,
        "memory_mode": memory_mode,
        "weight_mode": weight_mode,
        "use_injection": use_injection,
        "seed": seed,
        "rounds": rounds,
        "probe_size": probe_size,
        "git_commit": git_sha(),
        "hash0": hash0,
        "hash_final": param_hash(net),
        "param_norm_0": theta_norm(net),
        "param_norm_final": theta_norm(net),
        "net_drift": _nd(net, theta0),
        "gross_drift": gross,
        # core metrics
        "failure_rate": total_failures / max(rounds, 1),
        "correction_rate": total_corrections / max(total_failures, 1),
        "recovery_rate": total_recovery / max(total_failures, 1),
        # v0.5.1 separated recovery
        "CAR": recovery_metrics.correction_application_rate,
        "SRR": recovery_metrics.successful_recovery_rate,
        "RE": recovery_metrics.recovery_efficiency,
        # RFR variants
        "RFR_target": rfr_target,
        "RFR_similar": rfr_similar,
        "RFR_exact": rfr_exact,
        # weight adaptation
        "w_after_failure": float(w_after_failure),
        "w_after_success": float(w_after_success),
        "weight_adaptation": float(w_after_success - w_after_failure),
        # natural failure
        "natural_failure_rate": natural_detector.natural_failure_rate,
        "natural_failure_count": natural_detector.total_natural_failures,
        # memory
        "memory_stats": memory.get_category_counts(),
        "target_transitions": target_transitions,
        "weight_stats_by_outcome": weight_stats,
        "wall_seconds": time.time() - t0,
    }
    return result, memory, round_logs


# ====================================================================== #
# Multi-Seed Orchestration                                                #
# ====================================================================== #

def run_experiment_group(
    group_name: str,
    seeds: List[int],
    rounds: int,
    config: dict,
    data: dict,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """Run one experiment group across multiple seeds, aggregate results."""
    print(f"\n{'='*60}")
    print(f"Experiment Group: {group_name}")
    print(f"  Seeds: {seeds}")
    print(f"  Rounds: {rounds}")
    print(f"  Config: {kwargs}")
    print(f"{'='*60}")

    results = []
    all_round_logs = {}
    for si, seed in enumerate(seeds):
        print(f"\n  seed {seed} ({si+1}/{len(seeds)}) ...")
        result, memory, round_logs = run_v051_experiment(
            seed, rounds, group_name, config, data, **kwargs)
        results.append(result)
        all_round_logs[seed] = round_logs

        # save per-seed results
        seed_dir = os.path.join(output_dir, "results", f"{group_name}_s{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        with open(os.path.join(seed_dir, "memory.json"), "w") as f:
            json.dump(memory.snapshot(), f, indent=2, default=str)
        with open(os.path.join(seed_dir, "round_log.json"), "w") as f:
            json.dump(round_logs, f, indent=1, default=str)

    # aggregate
    agg = {}
    numeric_keys = [
        "failure_rate", "correction_rate", "recovery_rate",
        "CAR", "SRR", "RE",
        "RFR_target", "RFR_similar", "RFR_exact",
        "w_after_failure", "w_after_success", "weight_adaptation",
        "net_drift", "gross_drift",
        "natural_failure_rate", "natural_failure_count",
    ]
    for key in numeric_keys:
        vals = [r[key] for r in results if key in r]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            # 95% CI
            if len(vals) >= 2:
                se = float(np.std(vals)) / np.sqrt(len(vals))
                agg[f"{key}_ci95"] = [float(np.mean(vals) - 1.96 * se),
                                       float(np.mean(vals) + 1.96 * se)]

    agg["seeds"] = len(seeds)
    agg["rounds"] = rounds
    agg["experiment"] = group_name

    with open(os.path.join(output_dir, "results", f"{group_name}_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n  {group_name} aggregated:")
    for key in ["failure_rate", "SRR", "RFR_similar", "w_after_failure", "w_after_success"]:
        m = agg.get(f"{key}_mean", 0.0)
        s = agg.get(f"{key}_std", 0.0)
        print(f"    {key}: {m:.4f} ± {s:.4f}")

    return agg


# ====================================================================== #
# Main                                                                    #
# ====================================================================== #

def main():
    ap = argparse.ArgumentParser(description="v0.5.1 experiments")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--experiments", default="A1,A2,A3,A4,A5,C0,C2,C3,C4,C5,NF,CF")
    ap.add_argument("--out", default=BASE_DIR)
    ap.add_argument("--probe-size", type=int, default=32)
    args = ap.parse_args()

    # load config
    import yaml
    config_path = "config/phase5_1_v051.yaml"
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    config["num_rounds"] = args.rounds
    config["num_seeds"] = args.seeds

    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "configs"), exist_ok=True)
    with open(os.path.join(args.out, "configs", "v051.yaml"), "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # prepare data once
    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    data = prepare_data(p5_cfg)

    seeds = [SEED_BASE + i for i in range(args.seeds)]
    experiments = [e.strip() for e in args.experiments.split(",")]
    all_summaries = {}

    # define experiment groups
    EXP_DEFS = {
        # Memory Ablation (§21) — need injection to produce failures
        "A1": {"correction_mode": "memory_conditioned", "memory_mode": "full",
               "use_injection": True, "desc": "Full (C5+Memory) + Injection"},
        "A2": {"correction_mode": "error_conditioned", "memory_mode": "zero",
               "use_injection": True, "desc": "No Memory (C4) + Injection"},
        "A3": {"correction_mode": "memory_conditioned", "memory_mode": "shuffled",
               "use_injection": True, "desc": "Shuffled Memory + Injection"},
        "A4": {"correction_mode": "memory_conditioned", "memory_mode": "random",
               "use_injection": True, "desc": "Random Memory + Injection"},
        "A5": {"correction_mode": "memory_conditioned", "memory_mode": "zero",
               "use_injection": True, "desc": "Zero Memory + Injection"},
        # Correction Ablation (§23) — all need injection to produce failures
        "C0": {"correction_mode": "none", "memory_mode": "zero",
               "use_injection": True, "desc": "No Correction + Injection"},
        "C2": {"correction_mode": "reversal", "memory_mode": "zero",
               "use_injection": True, "desc": "Pure Reversal + Injection"},
        "C3": {"correction_mode": "learned", "memory_mode": "zero",
               "use_injection": True, "desc": "Learned (No Memory) + Injection"},
        "C4": {"correction_mode": "error_conditioned", "memory_mode": "zero",
               "use_injection": True, "desc": "Error-Conditioned + Injection"},
        "C5": {"correction_mode": "memory_conditioned", "memory_mode": "full",
               "use_injection": True, "desc": "Full Model + Injection"},
        # Natural Failure (§19)
        "NF": {"correction_mode": "memory_conditioned", "memory_mode": "full",
               "use_injection": False, "desc": "Natural Failure"},
        # Controlled Failure (§20)
        "CF": {"correction_mode": "memory_conditioned", "memory_mode": "full",
               "use_injection": True, "desc": "Controlled Failure + Memory"},
    }

    for exp in experiments:
        if exp not in EXP_DEFS:
            print(f"  [skip] unknown experiment: {exp}")
            continue
        defs = EXP_DEFS[exp]
        agg = run_experiment_group(
            exp, seeds, args.rounds, config, data, args.out,
            correction_mode=defs["correction_mode"],
            memory_mode=defs["memory_mode"],
            use_injection=defs["use_injection"],
            probe_size=args.probe_size,
        )
        all_summaries[exp] = agg

    # save all summaries
    with open(os.path.join(args.out, "results", "all_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    # final report
    print(f"\n{'='*70}")
    print("v0.5.1 Experiment Summary")
    print(f"{'='*70}")
    for exp, agg in all_summaries.items():
        desc = EXP_DEFS.get(exp, {}).get("desc", exp)
        srr = agg.get("SRR_mean", 0.0)
        rfr = agg.get("RFR_similar_mean", 0.0)
        wf = agg.get("w_after_failure_mean", 0.0)
        ws = agg.get("w_after_success_mean", 0.0)
        nfr = agg.get("natural_failure_rate_mean", 0.0)
        print(f"  {exp:4s} ({desc:30s}): SRR={srr:.3f} RFR_s={rfr:.3f} "
              f"w_f={wf:.3f} w_s={ws:.3f} NFR={nfr:.3f}")


if __name__ == "__main__":
    main()
