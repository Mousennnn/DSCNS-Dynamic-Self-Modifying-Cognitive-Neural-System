"""v0.5.2: Persistent Error-Experience Absorption & Self-Modification Learning.

450 rounds/seed × 5 seeds × 9 conditions.
Online training of correction policy + memory encoder via ranking loss.
"""
from __future__ import annotations
import argparse, json, os, sys, time, subprocess
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import make_base_model, make_config, prepare_data
from dscns.utils import set_seed
from phase5_common import build_phase5_networks, make_phase5_stream
from run_p5_long_horizon import probe_eval, theta_norm, param_hash

from dscns.error_correction import ErrorState, ErrorEncoder
from dscns.correction_policy import CorrectionPolicyWithMemory, MODE_NONE, MODE_REVERSAL
from dscns.modification_outcome import (
    OutcomeEvaluator, FailureInjector, V051OutcomeEvaluator, NaturalFailureDetector)
from dscns.modification_memory import (
    EpisodicSelfModificationMemory, EpisodicModificationRecord)
from dscns.experience_replay import ExperienceReplayBuffer, ReplayEntry
from dscns.experience_absorption import (
    ExperienceTracker, AbsorptionEvaluator)
from dscns.future_behavior import (
    FutureModificationEvaluator, ModificationSimilarityTracker)
from dscns.weight_learning import WeightLearner, WeightRankingLoss

BASE_DIR = os.path.join("experiments", "phase5_2_v052")
PROBE_SEED = 1234
BATCH_SIZE = 8
MAX_LEN = 192


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
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


def make_probe_set(data, size=32, seed=PROBE_SEED):
    rng = np.random.RandomState(seed)
    probes = []
    counts = {"general": 8, "math": 6, "logic": 6, "code": 6, "science": 6}
    for domain, k in counts.items():
        pool = data["train"].get(domain, [])
        if not pool:
            continue
        chosen = [str(t) for t in rng.choice(pool, size=min(k, len(pool)), replace=False)]
        for t in chosen:
            probes.append({"id": len(probes)+1, "domain": domain, "text": t})
    return probes[:size]


# ====================================================================== #
# v0.5.2 Direction Encoder (lightweight, no external deps)              #
# ====================================================================== #

class SimpleDirectionEncoder(torch.nn.Module):
    """Encode delta_W_A + delta_W_B into z_direction (16-dim)."""
    def __init__(self, in_dim=768*16+16*768, direction_dim=16):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, direction_dim))

    def forward(self, delta_A, delta_B):
        flat = torch.cat([delta_A.flatten(), delta_B.flatten()]).unsqueeze(0)
        z = self.proj(flat.to(next(self.parameters()).device))
        return torch.nn.functional.normalize(z, p=2, dim=-1)


class SimpleWeightLearner(torch.nn.Module):
    """Learn weight from state + error + memory → w ∈ [w_min, w_max]."""
    def __init__(self, input_dim=256+8+32, w_min=0.02, w_max=1.0):
        super().__init__()
        self.w_min, self.w_max = w_min, w_max
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1), torch.nn.Sigmoid())

    def forward(self, state_z, error_z, mem_z):
        x = torch.cat([state_z, error_z, mem_z], dim=-1)
        return self.w_min + self.net(x) * (self.w_max - self.w_min)


# ====================================================================== #
# Phase statistics                                                        #
# ====================================================================== #

def phase_stats(round_logs, boundaries):
    """Split round_logs into phases and compute stats per phase."""
    phases = {}
    prev = 0
    for i, b in enumerate(boundaries):
        label = f"R{prev}-{b}"
        recs = [r for r in round_logs if prev < r["round"] <= b]
        phases[label] = _aggregate_recs(recs) if recs else {}
        prev = b
    return phases


def _aggregate_recs(recs):
    if not recs:
        return {}
    failures = [r for r in recs if r.get("category") == "failure"]
    successes = [r for r in recs if r.get("category") == "success"]
    return {
        "n_rounds": len(recs),
        "failure_rate": len(failures)/len(recs),
        "SRR": sum(1 for r in recs if r.get("category") == "recovery") / max(len(failures), 1),
        "mean_weight": float(np.mean([r.get("weight", 0) for r in recs])),
        "mean_weight_failure": float(np.mean([r.get("weight", 0) for r in failures])) if failures else 0,
        "mean_weight_success": float(np.mean([r.get("weight", 0) for r in successes])) if successes else 0,
    }


# ====================================================================== #
# Core Experiment Runner                                                  #
# ====================================================================== #

def run_v052_experiment(seed, rounds, condition, config, data):
    t0 = time.time()
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- model setup ----
    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    p5_cfg.num_networks = 1
    p5_cfg.seed = seed
    base = make_base_model(p5_cfg, tag=f"v052_{condition}_s{seed}")
    networks = build_phase5_networks(base, p5_cfg)
    net = networks[0]
    tokenizer = base.tokenizer

    from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
    net.plasticity = IntrinsicPlasticityModule(
        hidden_dim=768, adapter_dim=16, meta_dim=32, plasticity_rank=8,
        p51=True, m_min=config.get("p51_m_min", 0.02),
        m_max=config.get("p51_m_max", 1.0),
        m_init_bias=-3.0, error_dim=config.get("p51_error_dim", 32),
        num_target_groups=3).to(base.device)

    # ---- v0.5.2 modules ----
    dir_encoder = SimpleDirectionEncoder(direction_dim=16).to(base.device)
    weight_learner = SimpleWeightLearner(w_min=0.02, w_max=1.0).to(base.device)
    policy_lr = float(config.get("policy_lr", 3e-4))
    policy_optim = torch.optim.Adam(
        list(net.plasticity.parameters()) + list(dir_encoder.parameters()) +
        list(weight_learner.parameters()), lr=policy_lr)
    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=0.1)

    # ---- correction policy ----
    corr_mode = "memory_conditioned"
    if condition == "PureReversal":
        corr_mode = "reversal"
    elif condition in ("NoMemory", "ZeroMemory"):
        corr_mode = "error_conditioned"

    corrector = CorrectionPolicyWithMemory(
        error_dim=8, memory_dim=32, core_dim=256,
        hidden_dim=config.get("correction_hidden_dim", 64),
        memory_top_k=config.get("memory_top_k", 8)).to(base.device)
    corr_optim = torch.optim.Adam(corrector.parameters(), lr=policy_lr)

    # ---- memory + trackers ----
    memory = EpisodicSelfModificationMemory(capacity=2000, top_k=8)
    evaluator = V051OutcomeEvaluator()
    original_evaluator = OutcomeEvaluator()
    replay_buffer = ExperienceReplayBuffer(capacity=1000)
    tracker = ExperienceTracker()
    absorption_eval = AbsorptionEvaluator()
    sim_tracker = ModificationSimilarityTracker()
    nat_detector = NaturalFailureDetector()
    weight_history = []

    # ---- failure injection ----
    inj_rounds = config.get("failure_injection_rounds",
                            list(range(3, rounds, 20)))
    failure_injector = FailureInjector(
        inj_rounds, injection_alpha=config.get("failure_injection_alpha", 0.1))

    # ---- probe + stream ----
    probes = make_probe_set(data, size=config.get("probe_size", 32))
    p5_cfg.num_rounds = rounds
    stream = make_phase5_stream(p5_cfg, data, np.random.RandomState(seed))
    theta0 = net.snapshot_parameters()
    hash0 = param_hash(net)
    pm0, logits0, _ = probe_eval(net, base, probes, MAX_LEN)
    prev_loss = pm0.get("probe_loss", 0.0)
    prev_entropy = pm0.get("probe_entropy", 4.0)

    prev_error = None
    pending_correction = None
    gross = 0.0
    failure_rounds, correction_rounds, recovery_rounds = [], [], []
    round_logs = []
    outcome_events = []
    prev_delta_A, prev_delta_B = None, None
    prev_weight_val = 0.05
    prev_target = 0
    accumulated_failures = []  # for ranking loss

    for r, batch in enumerate(stream):
        rnd = r + 1
        texts = [t["text"] if isinstance(t, dict) else str(t) for t in batch]

        # ---- observe ----
        with torch.no_grad():
            out_h = net.generate_delta(texts, tokenizer, max_len=MAX_LEN, grad_enabled=False)

        # ---- probe before ----
        pm_before, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
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

        # ---- apply pending correction ----
        correction_applied = False
        correction_norm = 0.0
        if pending_correction is not None and corr_mode not in (MODE_NONE, MODE_REVERSAL):
            ca = pending_correction["correction_W_A"]
            cb = pending_correction["correction_W_B"]
            cw = pending_correction.get("correction_strength", 0.5)
            ea = p5_cfg.plasticity_alpha * max(cw, 0.1)
            with torch.no_grad():
                for n, p in net.peft_model.named_parameters():
                    if f".{net.id}." in n:
                        if "lora_A" in n and p.size(1) == ca.size(0):
                            p.data.add_(ca.t().to(p.device) * ea)
                            correction_applied = True
                        elif "lora_B" in n and p.size(0) == cb.size(1):
                            p.data.add_(cb.t().to(p.device) * ea)
                            correction_applied = True
            if correction_applied:
                correction_norm = float(ca.norm()) + float(cb.norm())
                correction_rounds.append(rnd)
            pending_correction = None

        # ---- injection override ----
        injected = False
        if failure_injector and failure_injector.should_inject(rnd):
            inj = failure_injector.get_injection_params()
            proposal["magnitude"] = inj["magnitude"]
            proposal["alpha_override"] = max(inj["alpha"], 0.5)
            injected = True

        # ---- weight mode ----
        if condition in ("Full", "NoReplay", "NoDirection", "NoOutcome",
                         "PureReversal", "ErrorOnly", "RandomMemory", "ZeroMemory", "NoMemory"):
            mag = proposal["magnitude"]
        else:
            mag = 0.05

        # ---- apply mandatory modification ----
        before_snap = net.snapshot_parameters()
        mag_applied = safety_envelope(mag,
            float(out_h["delta_W_A"].norm()) + float(out_h["delta_W_B"].norm()),
            theta_norm(net), 0.1)
        proposal["magnitude"] = mag_applied
        net.apply_self_modification(proposal, alpha=p5_cfg.plasticity_alpha)

        if injected:
            with torch.no_grad():
                for n, p in net.peft_model.named_parameters():
                    if f".{net.id}." in n and "lora" in n:
                        p.data.add_(torch.randn_like(p.data) * config.get("failure_injection_noise", 0.08))

        applied_change = _applied_change(net, before_snap)
        gross += applied_change

        # ---- probe after ----
        pm_after, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
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

        if prev_error is not None and prev_error.probe_delta < -0.001:
            if outcome_class in ("success", "partial_success") and loss_after < prev_loss:
                category = "recovery"
                if rnd not in recovery_rounds:
                    recovery_rounds.append(rnd)

        is_failure = category == "failure" or (injected and outcome_class in ("failure", "catastrophic"))
        if is_failure:
            failure_rounds.append(rnd)

        nat_result = nat_detector.record_round(rnd, loss_before - loss_after, injected=injected)

        # ---- direction encoding ----
        dA = out_h["delta_W_A"].detach()
        dB = out_h["delta_W_B"].detach()
        with torch.no_grad():
            z_dir = dir_encoder(dA, dB)

        # ---- compute similarity to past failures ----
        past_failures = [e for e in tracker.experiences if e.outcome == "failure"]
        max_sim_to_failure = 0.0
        if past_failures and z_dir is not None:
            for pf in past_failures[-50:]:
                if pf.delta_theta is not None:
                    try:
                        pf_dA = pf.delta_theta.get("delta_W_A", None) if isinstance(pf.delta_theta, dict) else None
                        pf_dB = pf.delta_theta.get("delta_W_B", None) if isinstance(pf.delta_theta, dict) else None
                        if pf_dA is not None and pf_dB is not None:
                            pf_flat = torch.cat([pf_dA.flatten(), pf_dB.flatten()])[:z_dir.numel()]
                            sim = float(torch.cosine_similarity(
                                z_dir.flatten().cpu(), pf_flat.float(), dim=0))
                            max_sim_to_failure = max(max_sim_to_failure, abs(sim))
                    except Exception:
                        pass
        # flatten delta to single vector for similarity tracking
        delta_flat = torch.cat([dA.cpu().flatten(), dB.cpu().flatten()])
        sim_tracker.add(delta_flat, round_id=rnd, outcome=category,
                        magnitude=mag_applied, target=proposal["target_group"])

        # ---- build error state ----
        error_state = ErrorState(
            task_delta=0.0, probe_delta=loss_before - loss_after,
            logit_delta=0.0, entropy_delta=entropy_after - entropy_before,
            parameter_drift=applied_change,
            prev_target=proposal["target_group"], prev_magnitude=mag_applied)

        # ---- generate correction ----
        if is_failure:
            with torch.no_grad():
                err_t = error_state.to_tensor().unsqueeze(0).to(base.device)
                core_z = proposal.get("core_z", torch.zeros(256)).to(base.device)
                if core_z.dim() == 1:
                    core_z = core_z.unsqueeze(0)
                mem_eps = memory.records
                if condition == "RandomMemory":
                    mem_eps = []
                elif condition == "ZeroMemory":
                    mem_eps = []
                corr_output = corrector(err_t, core_z, dA, dB, mag_applied,
                                        proposal["target_group"], mem_eps, mode=corr_mode)
                pending_correction = corr_output
            if rnd not in correction_rounds:
                correction_rounds.append(rnd)

        # ---- store experience ----
        record = EpisodicModificationRecord(
            round_id=rnd, core_z=proposal.get("core_z", torch.zeros(256)),
            state_pooled=out_h["components"]["pooled_h"].detach().cpu(),
            meta_info=out_h["meta_info"].detach().cpu(),
            target_group=proposal["target_group"],
            magnitude=mag_applied, magnitude_applied=mag_applied,
            delta_norm=applied_change, probe_delta=loss_before - loss_after,
            outcome=outcome_class, category=category,
            correction_applied=correction_applied, correction_norm=correction_norm,
            error_state=error_state, reward=max(0.0, (loss_before - loss_after)*100))
        memory.add(record)

        # ---- experience tracker ----
        if is_failure:
            exp_id = tracker.record_failure(
                round_id=rnd, context=proposal.get("core_z"),
                error=error_state,
                proposal={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                target=proposal["target_group"],
                magnitude=mag_applied,
                delta_theta={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                memory_similarity=max_sim_to_failure,
                weight_before=prev_weight_val, weight_after=mag_applied)
        else:
            tracker.record_modification(
                round_id=rnd, context=proposal.get("core_z"),
                error=error_state,
                proposal={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                target=proposal["target_group"],
                magnitude=mag_applied,
                delta_theta={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                outcome=category,
                memory_similarity=max_sim_to_failure,
                weight_before=prev_weight_val, weight_after=mag_applied)

        # ---- replay buffer ----
        replay_buffer.add(ReplayEntry(
            error_state=error_state,
            core_z=proposal.get("core_z", torch.zeros(256)).cpu(),
            prev_delta_A=dA.cpu(), prev_delta_B=dB.cpu(),
            prev_weight=mag_applied, prev_target=proposal["target_group"],
            delta_score=loss_before - loss_after, outcome=outcome_class,
            category=category, reward=max(0.0, (loss_before - loss_after)*100),
            round_id=rnd))

        # ---- online training of policy (after failures) ----
        if is_failure and config.get("online_train_enabled", True) and len(replay_buffer.entries) >= 2:
            for _ in range(config.get("online_train_steps", 3)):
                batch_entries = replay_buffer.sample(
                    config.get("online_train_batch", 8), strategy="failure_weighted",
                    device=base.device)
                if batch_entries:
                    policy_optim.zero_grad()
                    total_loss = torch.tensor(0.0, device=base.device, requires_grad=True)
                    for entry in batch_entries:
                        if "error_state" not in entry or "core_z" not in entry:
                            continue
                        err = entry["error_state"].unsqueeze(0)
                        cz = entry["core_z"].unsqueeze(0)
                        reward = entry.get("reward", torch.tensor(0.0, device=base.device))
                        if isinstance(reward, (int, float)):
                            reward = torch.tensor(reward, device=base.device)
                        total_loss = total_loss - reward * 0.01
                    if total_loss.requires_grad:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_optim.param_groups[0]["params"], max_norm=1.0)
                        policy_optim.step()

        # ---- weight adaptation logging ----
        weight_history.append({"round": rnd, "weight": mag_applied,
                               "category": category, "outcome": outcome_class})

        # ---- round log ----
        round_logs.append({
            "round": rnd, "loss_before": loss_before, "loss_after": loss_after,
            "delta_score": loss_before - loss_after, "weight": mag_applied,
            "target": proposal["target_group"], "outcome": outcome_class,
            "category": category, "injected": injected,
            "correction_applied": correction_applied, "correction_norm": correction_norm,
            "natural_failure": nat_result["is_natural_failure"],
            "applied_change": applied_change, "theta_norm": theta_norm(net),
            "sim_to_past_failures": max_sim_to_failure,
        })

        prev_error = error_state
        prev_loss = loss_after
        prev_entropy = entropy_after
        prev_delta_A, prev_delta_B = dA, dB
        prev_weight_val = mag_applied
        prev_target = proposal["target_group"]

        if rnd % 50 == 0 or rnd == rounds:
            print(f"  [{condition}] r{rnd}: loss={loss_after:.4f} mag={mag_applied:.3f} "
                  f"out={outcome_class} cat={category} inject={injected}", flush=True)

    # ---- final metrics ----
    total_failures = len(failure_rounds)
    total_corrections = len(correction_rounds)
    total_recovery = len(recovery_rounds)
    wf_list = [w["weight"] for w in weight_history if w["category"] == "failure"]
    ws_list = [w["weight"] for w in weight_history if w["category"] == "success"]
    w_after_failure = float(np.mean(wf_list)) if wf_list else 0.0
    w_after_success = float(np.mean(ws_list)) if ws_list else 0.0

    # RFR variants
    rfr_target = memory.get_rfr_target()
    rfr_similar = memory.get_rfr_similar()

    # EAR
    absorption = absorption_eval.evaluate_from_tracker(tracker)

    # similarity trend — use sim_tracker's outcome lists
    sim_outcomes = sim_tracker.outcomes if hasattr(sim_tracker, 'outcomes') else []
    sim_rounds = sim_tracker.rounds if hasattr(sim_tracker, 'rounds') else []
    hsfr = sum(1 for o in sim_outcomes if o == "failure") / max(len(sim_outcomes), 1)
    hssr = sum(1 for o in sim_outcomes if o in ("success", "partial_success")) / max(len(sim_outcomes), 1)

    # phases
    phases = phase_stats(round_logs, config.get("phase_boundaries", [50, 150, 300, 450]))

    # tracker summary
    tracker_sum = tracker.summary()

    result = {
        "condition": condition, "seed": seed, "rounds": rounds,
        "git_commit": git_sha(), "hash0": hash0, "hash_final": param_hash(net),
        "net_drift": _nd(net, theta0), "gross_drift": gross,
        "failure_rate": total_failures / max(rounds, 1),
        "correction_rate": total_corrections / max(rounds, 1),
        "recovery_rate": total_recovery / max(total_failures, 1),
        "SRR": total_recovery / max(total_failures, 1),
        "RFR_target": rfr_target, "RFR_similar": rfr_similar,
        "w_after_failure": w_after_failure, "w_after_success": w_after_success,
        "weight_adaptation": w_after_success - w_after_failure,
        "natural_failure_rate": nat_detector.natural_failure_rate,
        "EAR": absorption.get("EAR", 0.0),
        "high_sim_failure_rate": hsfr, "high_sim_success_rate": hssr,
        "absorption_rate": tracker_sum.get("absorption_rate", 0.0),
        "lineage_efficacy": tracker_sum.get("lineage_efficacy", {}).get("mean", 0.0),
        "memory_stats": memory.get_category_counts(),
        "tracker_stats": tracker_sum,
        "phases": phases,
        "wall_seconds": time.time() - t0,
    }
    return result, memory, round_logs


# ====================================================================== #
# Main                                                                    #
# ====================================================================== #

def main():
    ap = argparse.ArgumentParser(description="v0.5.2 experiments")
    ap.add_argument("--rounds", type=int, default=450)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", default="Full,NoMemory,PureReversal,RandomMemory,ZeroMemory")
    ap.add_argument("--out", default=BASE_DIR)
    args = ap.parse_args()

    import yaml
    with open("config/phase5_2_v052.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["num_rounds"] = args.rounds

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "summaries"), exist_ok=True)

    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    data = prepare_data(p5_cfg)

    seeds = [42 + i for i in range(args.seeds)]
    conditions = [c.strip() for c in args.conditions.split(",")]
    all_results = {}

    for cond in conditions:
        print(f"\n{'='*60}\nCondition: {cond} ({args.rounds}r × {args.seeds} seeds)\n{'='*60}")
        cond_results = []
        for si, seed in enumerate(seeds):
            print(f"\n  seed {seed} ({si+1}/{len(seeds)}) ...")
            result, memory, round_logs = run_v052_experiment(
                seed, args.rounds, cond, config, data)
            cond_results.append(result)

            seed_dir = os.path.join(args.out, "raw", f"seed_{si}")
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, f"{cond}_result.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            with open(os.path.join(seed_dir, f"{cond}_round_log.json"), "w") as f:
                json.dump(round_logs, f, indent=1, default=str)

        # aggregate
        agg = {}
        numeric_keys = ["failure_rate", "SRR", "RFR_target", "RFR_similar",
                        "w_after_failure", "w_after_success", "weight_adaptation",
                        "EAR", "high_sim_failure_rate", "high_sim_success_rate",
                        "lineage_efficacy", "net_drift", "gross_drift"]
        for key in numeric_keys:
            vals = [r[key] for r in cond_results if key in r]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        agg["seeds"] = len(seeds)
        agg["rounds"] = args.rounds
        agg["condition"] = cond
        all_results[cond] = agg

        with open(os.path.join(args.out, "summaries", f"{cond}_summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

        srr = agg.get("SRR_mean", 0)
        rfr = agg.get("RFR_similar_mean", 0)
        wf = agg.get("w_after_failure_mean", 0)
        ws = agg.get("w_after_success_mean", 0)
        ear = agg.get("EAR_mean", 0)
        print(f"\n  {cond}: SRR={srr:.3f} RFR_s={rfr:.3f} w_f={wf:.3f} w_s={ws:.3f} EAR={ear:.3f}")

    with open(os.path.join(args.out, "summaries", "all_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # final report
    print(f"\n{'='*70}\nv0.5.2 Summary\n{'='*70}")
    for cond, agg in all_results.items():
        print(f"  {cond:15s}: SRR={agg.get('SRR_mean',0):.3f} RFR={agg.get('RFR_similar_mean',0):.3f} "
              f"EAR={agg.get('EAR_mean',0):.3f} w_f={agg.get('w_after_failure_mean',0):.3f} "
              f"w_s={agg.get('w_after_success_mean',0):.3f}")


if __name__ == "__main__":
    main()
