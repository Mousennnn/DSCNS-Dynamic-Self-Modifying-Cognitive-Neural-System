"""v0.5.3: Persistent Experience-to-Policy Learning (Phase 5.5).

450 rounds/seed × 5 seeds × 8 conditions.
Experience → Credit → Value → Policy → Modification.

Key difference from v0.5.2:  Memory directly conditions the Modification
Policy (via PolicyAdapter), not just the Correction signal.
"""
from __future__ import annotations
import argparse, json, os, sys, time, subprocess, hashlib
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

# ---- v0.5.3 new modules ----
from dscns.experience_credit import ExperienceCreditAssigner, TemporalCreditTracker
from dscns.experience_value import ExperienceValueModel
from dscns.policy_adapter import PolicyAdapter
from dscns.policy_learning import ModificationPolicyLearner
from dscns.alternative_proposal import AlternativeProposalGenerator, ModificationCandidate

BASE_DIR = os.path.join("experiments", "phase5_3_v053")
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
    return min(magnitude, max_ratio * param_norm / max(delta_norm, 1e-6))


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
# Lightweight direction encoder (from v0.5.2)                            #
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


# ====================================================================== #
# Simple weight learner (from v0.5.2)                                    #
# ====================================================================== #

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
# Policy divergence metrics                                               #
# ====================================================================== #

def compute_policy_divergence(
    policy_A_dist: List[Dict[int, float]],
    policy_B_dist: List[Dict[int, float]],
) -> Dict[str, float]:
    """Compute KL, JS divergence, and cosine similarity between two
    policy target distributions collected over time.

    Args:
        policy_A_dist: list of {target: prob} dicts from condition A
        policy_B_dist: list of {target: prob} dicts from condition B

    Returns:
        Dict with kl, js, cosine, n_samples.
    """
    if not policy_A_dist or not policy_B_dist:
        return {"kl": 0.0, "js": 0.0, "cosine": 0.0, "n_samples": 0}

    # aggregate into mean distributions
    n_targets = 3
    def _mean_dist(dists):
        counts = np.zeros(n_targets)
        for d in dists:
            for t, p in d.items():
                counts[min(int(t), n_targets-1)] += p
        counts /= max(len(dists), 1)
        total = counts.sum()
        if total > 0:
            counts /= total
        else:
            counts = np.ones(n_targets) / n_targets
        return counts

    p = _mean_dist(policy_A_dist)
    q = _mean_dist(policy_B_dist)

    # add epsilon for numerical stability
    eps = 1e-8
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()

    # KL(p || q)
    kl = float(np.sum(p * np.log(p / q)))

    # JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5*(p+q)
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    # cosine similarity
    norm = np.linalg.norm(p) * np.linalg.norm(q)
    cosine = float(np.dot(p, q) / max(norm, 1e-12))

    return {"kl": kl, "js": js, "cosine": cosine,
            "n_samples": len(policy_A_dist)}


# ====================================================================== #
# Phase statistics                                                        #
# ====================================================================== #

def phase_stats(round_logs, boundaries):
    phases = {}
    prev = 0
    for b in boundaries:
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
        "mean_target": float(np.mean([r.get("target", 0) for r in recs])),
    }


# ====================================================================== #
# Core Experiment Runner                                                  #
# ====================================================================== #

def run_v053_experiment(seed, rounds, condition, config, data):
    t0 = time.time()
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_str = config.get("device", "cuda")

    # ---- model setup ----
    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    p5_cfg.num_networks = 1
    p5_cfg.seed = seed
    base = make_base_model(p5_cfg, tag=f"v053_{condition}_s{seed}")
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

    # ---- v0.5.3 modules ----
    dir_encoder = SimpleDirectionEncoder(direction_dim=16).to(base.device)
    weight_learner = SimpleWeightLearner(w_min=0.02, w_max=1.0).to(base.device)

    # Policy adapter: the KEY new module
    policy_adapter = PolicyAdapter(
        state_dim=256, error_dim=32, memory_dim=32, value_dim=16,
        hidden_dim=config.get("policy_adapter_hidden", 256),
        n_candidates=config.get("policy_adapter_n_candidates", 4),
        m_min=config.get("p51_m_min", 0.02),
        m_max=config.get("p51_m_max", 1.0),
        n_target_groups=3).to(base.device)

    # Credit assignment
    credit_assigner = ExperienceCreditAssigner(
        gamma=config.get("credit_gamma", 0.95),
        default_k=config.get("credit_default_k", 3))
    credit_tracker = TemporalCreditTracker()

    # Experience value
    exp_value_model = ExperienceValueModel(
        learning_rate=config.get("exp_value_learning_rate", 0.1),
        decay_rate=config.get("exp_value_decay_rate", 0.001),
        capacity=config.get("exp_value_capacity", 5000))

    # Policy learning
    policy_learner = ModificationPolicyLearner(
        lr=config.get("policy_learning_rate", 3e-4),
        lambda_contrastive=config.get("lambda_contrastive", 1.0),
        lambda_avoid=config.get("lambda_avoid", 0.5),
        lambda_reuse=config.get("lambda_reuse", 0.5),
        lambda_stability=config.get("lambda_stability", 0.1),
        contrastive_margin=config.get("contrastive_margin", 0.1),
        device=base.device)

    # Alternative proposals
    alt_generator = AlternativeProposalGenerator(
        n_candidates=config.get("policy_adapter_n_candidates", 4),
        n_target_groups=3,
        m_min=config.get("p51_m_min", 0.02),
        m_max=config.get("p51_m_max", 1.0),
        exploration_eps=config.get("exploration_eps", 0.15),
        exploration_min=config.get("exploration_min", 0.02),
        exploration_decay=config.get("exploration_decay", 0.001))

    # Optimizers
    policy_lr = float(config.get("policy_learning_rate", 3e-4))
    policy_optim = torch.optim.Adam(
        list(net.plasticity.parameters()) + list(dir_encoder.parameters()) +
        list(weight_learner.parameters()) + list(policy_adapter.parameters()),
        lr=policy_lr)

    # ---- correction policy (from v0.5.2) ----
    corr_mode = "memory_conditioned"
    if condition == "NoMemory":
        corr_mode = "error_conditioned"

    corrector = CorrectionPolicyWithMemory(
        error_dim=8, memory_dim=32, core_dim=256,
        hidden_dim=config.get("correction_hidden_dim", 64),
        memory_top_k=config.get("memory_top_k", 8)).to(base.device)

    # ---- memory + trackers ----
    memory = EpisodicSelfModificationMemory(capacity=2000, top_k=8)
    evaluator = V051OutcomeEvaluator()
    original_evaluator = OutcomeEvaluator()
    replay_buffer = ExperienceReplayBuffer(capacity=1000)
    tracker = ExperienceTracker()
    absorption_eval = AbsorptionEvaluator()
    sim_tracker = ModificationSimilarityTracker()
    nat_detector = NaturalFailureDetector()

    # ---- v0.5.3: policy distribution history (for D_policy) ----
    policy_target_history: List[Dict[int, float]] = []  # per-round target probs
    policy_checkpoints: Dict[int, Dict] = {}

    # ---- failure injection ----
    inj_rounds = config.get("failure_injection_rounds", list(range(3, rounds, 20)))
    failure_injector = FailureInjector(
        inj_rounds, injection_alpha=config.get("failure_injection_alpha", 0.1))

    # ---- probe + stream ----
    probes = make_probe_set(data, size=config.get("probe_size", 32))
    p5_cfg.num_rounds = 20
    base_stream = make_phase5_stream(p5_cfg, data, np.random.RandomState(seed))
    stream = []
    while len(stream) < rounds:
        stream.extend(base_stream)
    stream = stream[:rounds]
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
    weight_history = []
    exp_id_counter = 0

    # ---- condition-specific flags ----
    use_memory = condition not in ("NoMemory",)
    use_credit = condition not in ("NoCredit",)
    use_alternatives = condition not in ("NoAlternatives",)
    use_exploration = condition not in ("NoExploration",)
    freeze_policy = condition == "FrozenPolicy"

    # exploration rate override
    if not use_exploration:
        exploration_eps = 0.0
    else:
        exploration_eps = config.get("exploration_eps", 0.15)

    for r, batch in enumerate(stream):
        rnd = r + 1
        texts = [t["text"] if isinstance(t, dict) else str(t) for t in batch]

        # ---- observe ----
        with torch.no_grad():
            out_h = net.generate_delta(texts, tokenizer, max_len=MAX_LEN, grad_enabled=False)

        # ---- probe before (every 5 rounds for speed) ----
        do_probe = (rnd % 5 == 1) or (rnd == 1) or (rnd == rounds)
        if do_probe:
            pm_before, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
            loss_before = pm_before.get("probe_loss", 0.0)
            entropy_before = pm_before.get("probe_entropy", 4.0)
        else:
            loss_before = prev_loss
            entropy_before = prev_entropy
        score_before = -loss_before

        # ---- compute experience value for current context ----
        # Create a simple value embedding from recent experience values
        recent_values = list(exp_value_model.values.values())[-10:]
        if recent_values:
            vals = [ev.value for ev in recent_values]
            mean_val = float(np.mean(vals))
            val_z = torch.tensor(
                [mean_val] * 8 + [0.0] * 8,  # 16-dim value embedding
                dtype=torch.float32, device=base.device).unsqueeze(0)
        else:
            val_z = torch.zeros(1, config.get("policy_adapter_value_dim", 16),
                                device=base.device)

        # ---- retrieve memory ----
        # Use pooled hidden as core_z (768-dim), project to 256 for corrector
        core_z_for_memory = out_h["components"]["pooled_h"].detach().to(base.device)
        if core_z_for_memory.dim() == 3:
            core_z_for_memory = core_z_for_memory.mean(1)
        core_z_flat_768 = core_z_for_memory.mean(0, keepdim=True)  # (1, 768)
        core_z_flat = core_z_flat_768[:, :256]  # (1, 256) for policy adapter

        # Retrieve similar episodes from memory
        mem_eps = memory.records if use_memory else []
        if condition == "RandomMemory":
            np.random.shuffle(mem_eps)
            mem_eps = mem_eps[:config.get("memory_top_k", 8)]
        elif condition == "ZeroMemory":
            mem_eps = []

        # Build memory embedding for policy adapter
        if mem_eps:
            mem_features = []
            for ep in mem_eps[-config.get("memory_top_k", 8):]:
                target_oh = torch.zeros(3, device=base.device)
                target_oh[min(getattr(ep, "target_group", 0), 2)] = 1.0
                err = torch.zeros(8, device=base.device)
                if hasattr(ep, "error_state") and ep.error_state is not None:
                    if hasattr(ep.error_state, "to_tensor"):
                        err = ep.error_state.to_tensor().to(base.device)
                feat = torch.cat([
                    torch.tensor([getattr(ep, "delta_norm", 0.0),
                                  getattr(ep, "magnitude", 0.0)], device=base.device),
                    target_oh,
                    torch.tensor([getattr(ep, "probe_delta", 0.0)], device=base.device),
                    err,
                ])
                mem_features.append(feat)
            if mem_features:
                mem_z_tensor = torch.stack(mem_features).mean(0, keepdim=True)  # (1, mem_dim)
                # pad/truncate to 32
                if mem_z_tensor.size(-1) < 32:
                    mem_z_tensor = torch.cat([
                        mem_z_tensor,
                        torch.zeros(1, 32 - mem_z_tensor.size(-1), device=base.device)
                    ], dim=-1)
                mem_z_tensor = mem_z_tensor[:, :32]
            else:
                mem_z_tensor = torch.zeros(1, 32, device=base.device)
        else:
            mem_z_tensor = torch.zeros(1, 32, device=base.device)

        # ---- error state ----
        error_state = ErrorState(
            task_delta=0.0, probe_delta=loss_before - prev_loss if rnd > 1 else 0.0,
            logit_delta=0.0, entropy_delta=entropy_before - prev_entropy,
            parameter_drift=0.0, prev_target=0, prev_magnitude=0.0)
        err_t = error_state.to_tensor().unsqueeze(0).to(base.device)
        error_z = net.plasticity.error_encoder(err_t) if hasattr(net.plasticity, 'error_encoder') else torch.zeros(1, 32, device=base.device)

        # ---- generate base proposal from plasticity module ----
        with torch.no_grad():
            proposal = net.plasticity.generate_proposal(
                out_h["components"]["pooled_h"].unsqueeze(1),
                net._current_params_tensors(),
                net._get_meta_info(net.plasticity_cfg.get("meta_dim", 32)),
                error_state=prev_error, memory_z=None, mask=None)

        # ---- POLICY ADAPTER: experience-conditioned proposal ----
        if not freeze_policy and rnd > 1:
            with torch.no_grad():
                policy_out = policy_adapter(
                    core_z_flat.float(),     # (1, 256)
                    error_z.float(),         # (1, 32)
                    mem_z_tensor.float(),    # (1, 32)
                    val_z.float(),           # (1, 16)
                )
            # Override proposal with policy adapter's decisions
            target_probs = policy_out["target_probs"].squeeze(0).detach()
            policy_adapter_target = int(target_probs.argmax().item())
            policy_adapter_magnitude = float(policy_out["magnitude"].squeeze().item())

            # Store target distribution for D_policy
            policy_target_history.append({
                i: float(target_probs[i].item()) for i in range(3)
            })

            # Use policy adapter's magnitude
            proposal["magnitude"] = policy_adapter_magnitude
            proposal["target_group"] = policy_adapter_target
            proposal["target_probs"] = target_probs
        else:
            # first round or frozen: use base proposal
            target_probs = proposal.get("target_probs", torch.ones(3) / 3)
            if isinstance(target_probs, torch.Tensor):
                target_probs = target_probs.detach()
                policy_target_history.append(
                    {i: float(target_probs[i].item()) for i in range(3)})
            else:
                policy_target_history.append({0: 0.33, 1: 0.33, 2: 0.34})

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

        # ---- alternative proposal generation ----
        if use_alternatives and rnd > 5:
            candidates = alt_generator.generate_candidates(
                base_target=proposal["target_group"],
                base_magnitude=proposal["magnitude"],
                candidate_scores=policy_out["candidate_scores"] if not freeze_policy and rnd > 1 else None,
            )
            selected_candidate = alt_generator.select_candidate(
                candidates, exploration_eps=exploration_eps)
            # Use selected candidate
            proposal["target_group"] = selected_candidate.target_group
            proposal["magnitude"] = selected_candidate.magnitude
        else:
            selected_candidate = ModificationCandidate(
                candidate_id=0, target_group=proposal["target_group"],
                magnitude=proposal["magnitude"], score=0.0, selected=True)

        # ---- apply mandatory modification ----
        before_snap = net.snapshot_parameters()
        mag = proposal["magnitude"]
        delta_norm_val = float(out_h["delta_W_A"].norm()) + float(out_h["delta_W_B"].norm())
        mag_applied = safety_envelope(mag, delta_norm_val, theta_norm(net), 0.1)
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
        if do_probe:
            pm_after, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
            loss_after = pm_after.get("probe_loss", 0.0)
            entropy_after = pm_after.get("probe_entropy", 4.0)
        else:
            loss_after = loss_before
            entropy_after = entropy_before
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

        # ---- compute reward for credit assignment ----
        reward = (loss_before - loss_after) * 10.0  # positive if improvement
        reward = float(np.clip(reward, -1.0, 1.0))

        # ---- CREDIT ASSIGNMENT ----
        if use_credit:
            mod_info = {
                "target": proposal["target_group"],
                "magnitude": mag_applied,
                "direction_norm": delta_norm_val,
            }
            credit_assigner.record_reward(rnd, reward, mod_info)
            credit_signal = credit_assigner.compute_credit(rnd)
            credit_tracker.record(rnd, credit_signal, mod_info, category)
        else:
            # no credit: use immediate reward only
            from dscns.experience_credit import CreditSignal
            credit_signal = CreditSignal(
                round_id=rnd, immediate_reward=reward,
                discounted_return=reward, cumulative_credit=reward)

        # ---- EXPERIENCE VALUE ----
        exp_id_counter += 1
        exp_id = f"exp-{exp_id_counter:05d}"
        exp_value_model.record(
            experience_id=exp_id, round_id=rnd,
            reward=credit_signal.cumulative_credit,
            experience_type=category if category in ("success", "failure", "recovery") else "failure",
            target_group=proposal["target_group"],
            magnitude=mag_applied,
        )

        # ---- store experience in memory ----
        delta_flat = torch.cat([dA.cpu().flatten(), dB.cpu().flatten()])
        sim_tracker.add(delta_flat, round_id=rnd, outcome=category,
                        magnitude=mag_applied, target=proposal["target_group"])

        record = EpisodicModificationRecord(
            round_id=rnd, core_z=out_h["components"]["pooled_h"].detach().cpu().mean(0),
            state_pooled=out_h["components"]["pooled_h"].detach().cpu(),
            target_group=proposal["target_group"],
            magnitude=mag_applied, magnitude_applied=mag_applied,
            delta_norm=applied_change, probe_delta=loss_before - loss_after,
            outcome=outcome_class, category=category,
            correction_applied=correction_applied, correction_norm=correction_norm,
            error_state=error_state, reward=max(0.0, (loss_before - loss_after)*100))
        memory.add(record)

        # ---- experience tracker ----
        if is_failure:
            exp_id_tracker = tracker.record_failure(
                round_id=rnd, context=out_h["components"]["pooled_h"].detach().cpu().mean(0),
                error=error_state,
                proposal={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                target=proposal["target_group"], magnitude=mag_applied,
                delta_theta={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()})
        else:
            exp_id_tracker = tracker.record_modification(
                round_id=rnd, context=out_h["components"]["pooled_h"].detach().cpu().mean(0),
                error=error_state,
                proposal={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                target=proposal["target_group"], magnitude=mag_applied,
                delta_theta={"delta_W_A": dA.cpu(), "delta_W_B": dB.cpu()},
                outcome=category)

        # ---- alternative proposal outcome recording ----
        if use_alternatives and selected_candidate is not None:
            alt_generator.record_outcome(selected_candidate, category)

        # ---- GENERATION CORRECTION (for next round) ----
        if is_failure:
            with torch.no_grad():
                err_t_corr = error_state.to_tensor().unsqueeze(0).to(base.device)
                # core_z must be 256-dim for CorrectionPolicy.context_encoder(256→32)
                core_z_for_corr = proposal.get("core_z", torch.zeros(256)).to(base.device)
                if core_z_for_corr.dim() == 1:
                    core_z_for_corr = core_z_for_corr.unsqueeze(0)
                if core_z_for_corr.size(-1) != 256:
                    # project from pooled hidden (768) to 256
                    pooled_768 = out_h["components"]["pooled_h"].detach().to(base.device)
                    if pooled_768.dim() == 3:
                        pooled_768 = pooled_768.mean(1)
                    pooled_768 = pooled_768.mean(0, keepdim=True)
                    # truncate to 256 (simple projection)
                    core_z_for_corr = pooled_768[:, :256]
                corr_output = corrector(err_t_corr, core_z_for_corr, dA, dB,
                                        mag_applied, proposal["target_group"],
                                        mem_eps, mode=corr_mode)
                pending_correction = corr_output
            if rnd not in correction_rounds:
                correction_rounds.append(rnd)

        # ---- POLICY LEARNING ----
        if not freeze_policy and is_failure and config.get("online_train_enabled", True):
            # Compute policy loss
            if rnd > 1 and 'policy_out' in dir():
                # Gather failed targets for avoidance
                failed_targets_tensor = torch.tensor(
                    [proposal["target_group"]], device=base.device)

                # Gather successful targets for reuse
                success_records = [m for m in memory.records if m.category == "success"][-5:]
                if success_records:
                    success_targets_tensor = torch.tensor(
                        [m.target_group for m in success_records], device=base.device)
                else:
                    success_targets_tensor = None

                # Stability: compare with previous probs
                old_probs = None
                if len(policy_target_history) > 1:
                    prev_dist = policy_target_history[-2]
                    old_probs = torch.tensor(
                        [prev_dist.get(i, 1/3) for i in range(3)],
                        device=base.device).unsqueeze(0)

                # Compute loss
                new_probs = policy_out["target_probs"].detach()
                losses = policy_learner.compute_loss(
                    new_probs, category, proposal["target_group"],
                    old_target_probs=old_probs,
                    failed_targets=failed_targets_tensor,
                    success_targets=success_targets_tensor)

                # Backward and step (only if loss > 0)
                if losses["total"].requires_grad and float(losses["total"]) > 0:
                    policy_optim.zero_grad()
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy_optim.param_groups[0]["params"], max_norm=1.0)
                    policy_optim.step()

        # ---- experience value update ----
        # Verify past failure experiences in similar context
        if not is_failure and rnd > 10:
            # Find past failures that are similar and mark as verified
            for past_ev in exp_value_model.get_failure_values()[-5:]:
                if past_ev.experience_id != exp_id:
                    exp_value_model.verify(past_ev.experience_id, success=True)

        # ---- weight adaptation logging ----
        weight_history.append({"round": rnd, "weight": mag_applied,
                               "category": category, "outcome": outcome_class})

        # ---- policy checkpoint ----
        cp_rounds = config.get("checkpoint_rounds", [0, 50, 100, 200, 300, 450])
        if rnd in cp_rounds:
            policy_checkpoints[rnd] = {
                "target_distribution": policy_target_history[-1] if policy_target_history else {},
                "exploration_rate": exploration_eps,
                "round": rnd,
            }

        # ---- round log ----
        round_logs.append({
            "round": rnd, "loss_before": loss_before, "loss_after": loss_after,
            "delta_score": loss_before - loss_after, "weight": mag_applied,
            "target": proposal["target_group"], "outcome": outcome_class,
            "category": category, "injected": injected,
            "correction_applied": correction_applied, "correction_norm": correction_norm,
            "natural_failure": nat_result["is_natural_failure"],
            "applied_change": applied_change, "theta_norm": theta_norm(net),
            "credit": credit_signal.cumulative_credit,
            "experience_value": exp_value_model.get_value(exp_id),
            "candidate_selected": selected_candidate.candidate_id if selected_candidate else 0,
            "n_candidates": alt_generator.n_candidates if use_alternatives else 1,
            "exploration_eps": exploration_eps,
        })

        prev_error = error_state
        prev_loss = loss_after
        prev_entropy = entropy_after
        prev_weight_val = mag_applied

        if rnd % 50 == 0 or rnd == rounds:
            credit_stats = credit_assigner.credit_statistics()
            print(f"  [{condition}] r{rnd}: loss={loss_after:.4f} mag={mag_applied:.3f} "
                  f"out={outcome_class} cat={category} credit={credit_signal.cumulative_credit:.3f} "
                  f"exp_val={exp_value_model.get_value(exp_id):.3f}", flush=True)

    # ================================================================== #
    # Final metrics                                                       #
    # ================================================================== #
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

    # credit statistics
    credit_stats = credit_assigner.credit_statistics()

    # experience value statistics
    ev_stats = exp_value_model.summary()

    # policy learning statistics
    pl_stats = policy_learner.loss_statistics()

    # alternative proposal statistics
    alt_stats = alt_generator.summary()

    # phases
    phases = phase_stats(round_logs, config.get("phase_boundaries", [50, 150, 300, 450]))

    # tracker summary
    tracker_sum = tracker.summary()

    # high-similarity failure/success rates
    sim_outcomes = sim_tracker.outcomes if hasattr(sim_tracker, 'outcomes') else []
    hsfr = sum(1 for o in sim_outcomes if o == "failure") / max(len(sim_outcomes), 1)
    hssr = sum(1 for o in sim_outcomes if o in ("success", "partial_success")) / max(len(sim_outcomes), 1)

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
        # ---- v0.5.3 new metrics ----
        "credit_mean": credit_stats.get("mean_credit", 0.0),
        "credit_std": credit_stats.get("std_credit", 0.0),
        "experience_value_mean": ev_stats.get("mean_value", 0.0),
        "experience_value_std": ev_stats.get("std_value", 0.0),
        "n_experience_values": ev_stats.get("n_experiences", 0),
        "alt_success_rate": alt_stats.get("success_rate", 0.0),
        "alt_failure_rate": alt_stats.get("failure_rate", 0.0),
        "alt_n_selected": alt_stats.get("n_selected", 0),
        "exploration_rate": exploration_eps,
        # ---- memory stats ----
        "memory_stats": memory.get_category_counts(),
        "tracker_stats": tracker_sum,
        "credit_statistics": credit_stats,
        "experience_value_statistics": ev_stats,
        "alternative_proposal_statistics": alt_stats,
        "policy_learning_statistics": pl_stats,
        "phases": phases,
        "policy_checkpoints": {str(k): v for k, v in policy_checkpoints.items()},
        "policy_target_distribution_mean": (
            {str(k): float(np.mean([d.get(k, 1/3) for d in policy_target_history]))
             for k in range(3)}
            if policy_target_history else {}
        ),
        "wall_seconds": time.time() - t0,
    }
    return result, memory, round_logs, policy_target_history


# ====================================================================== #
# Main                                                                    #
# ====================================================================== #

def main():
    ap = argparse.ArgumentParser(description="v0.5.3 experiments")
    ap.add_argument("--rounds", type=int, default=450)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", default="FullPolicy,NoMemory,FrozenPolicy,RandomMemory,ZeroMemory,NoCredit,NoAlternatives,NoExploration")
    ap.add_argument("--out", default=BASE_DIR)
    args = ap.parse_args()

    import yaml
    with open("config/phase5_3_v053.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["num_rounds"] = args.rounds

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "summaries"), exist_ok=True)

    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    data = prepare_data(p5_cfg)

    seeds = [42 + i for i in range(args.seeds)]
    conditions = [c.strip() for c in args.conditions.split(",")]
    all_results = {}
    all_policy_distributions = {}  # for cross-condition D_policy

    for cond in conditions:
        print(f"\n{'='*60}\nCondition: {cond} ({args.rounds}r × {args.seeds} seeds)\n{'='*60}")
        cond_results = []
        cond_policy_dists = []
        for si, seed in enumerate(seeds):
            print(f"\n  seed {seed} ({si+1}/{len(seeds)}) ...")
            result, memory, round_logs, policy_dist = run_v053_experiment(
                seed, args.rounds, cond, config, data)
            cond_results.append(result)
            cond_policy_dists.append(policy_dist)

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
                        "lineage_efficacy", "net_drift", "gross_drift",
                        "credit_mean", "credit_std",
                        "experience_value_mean", "experience_value_std",
                        "alt_success_rate", "alt_failure_rate"]
        for key in numeric_keys:
            vals = [r[key] for r in cond_results if key in r]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        agg["seeds"] = len(seeds)
        agg["rounds"] = args.rounds
        agg["condition"] = cond
        all_results[cond] = agg
        all_policy_distributions[cond] = cond_policy_dists

        with open(os.path.join(args.out, "summaries", f"{cond}_summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

        srr = agg.get("SRR_mean", 0)
        rfr = agg.get("RFR_similar_mean", 0)
        ear = agg.get("EAR_mean", 0)
        credit_m = agg.get("credit_mean", 0)
        print(f"\n  {cond}: SRR={srr:.3f} RFR={rfr:.3f} EAR={ear:.3f} credit={credit_m:.3f}")

    # ---- Cross-condition D_policy ----
    print(f"\n{'='*60}\nPolicy Divergence Analysis\n{'='*60}")
    policy_div_results = {}
    baseline_cond = "FullPolicy"
    if baseline_cond in all_policy_distributions:
        baseline_dists = all_policy_distributions[baseline_cond]
        # average over seeds
        avg_baseline = []
        if baseline_dists:
            max_len = max(len(d) for d in baseline_dists)
            for i in range(max_len):
                avg_dist = {}
                for d in baseline_dists:
                    if i < len(d):
                        for k, v in d[i].items():
                            avg_dist[k] = avg_dist.get(k, 0) + v / len(baseline_dists)
                if avg_dist:
                    avg_baseline.append(avg_dist)

        for cond in conditions:
            if cond == baseline_cond:
                continue
            if cond in all_policy_distributions:
                cond_dists = all_policy_distributions[cond]
                avg_cond = []
                if cond_dists:
                    max_len = max(len(d) for d in cond_dists)
                    for i in range(max_len):
                        avg_dist = {}
                        for d in cond_dists:
                            if i < len(d):
                                for k, v in d[i].items():
                                    avg_dist[k] = avg_dist.get(k, 0) + v / len(cond_dists)
                        if avg_dist:
                            avg_cond.append(avg_dist)

                div = compute_policy_divergence(avg_baseline, avg_cond)
                policy_div_results[f"{baseline_cond}_vs_{cond}"] = div
                print(f"  D_policy({baseline_cond} vs {cond}): "
                      f"KL={div['kl']:.4f} JS={div['js']:.4f} Cos={div['cosine']:.4f}")

    # Save D_policy results
    with open(os.path.join(args.out, "summaries", "policy_divergence.json"), "w") as f:
        json.dump(policy_div_results, f, indent=2)

    # Save all summary
    all_results["_policy_divergence"] = policy_div_results
    with open(os.path.join(args.out, "summaries", "all_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # final report
    print(f"\n{'='*70}\nv0.5.3 Summary\n{'='*70}")
    for cond, agg in all_results.items():
        if cond.startswith("_"):
            continue
        print(f"  {cond:15s}: SRR={agg.get('SRR_mean',0):.3f} RFR={agg.get('RFR_similar_mean',0):.3f} "
              f"EAR={agg.get('EAR_mean',0):.3f} credit={agg.get('credit_mean',0):.3f} "
              f"exp_val={agg.get('experience_value_mean',0):.3f}")

    print(f"\n  Policy Divergence:")
    for pair, div in policy_div_results.items():
        print(f"    {pair}: KL={div['kl']:.4f} JS={div['js']:.4f}")


if __name__ == "__main__":
    main()
