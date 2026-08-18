"""Shared helpers for the Phase 5 (intrinsic self-modification) experiments.

The P5 loop is deliberately lean (report section 9.1): each cognitive
network runs standard task learning on the round's experiences and, at an
external fixed frequency, runs one intrinsic plasticity step:

    snapshot -> generate delta (model mechanism) -> apply (model method)
             -> quick validation (experiment controller) -> accept / rollback

Negative-control deltas (random / constant / shuffled) are injected through
the same step so the only difference is the delta content.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dscns.config import DSCNSConfig
from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
from dscns.memory import MemorySystem
from dscns.networks import CognitiveNetwork

DOMAINS = ["general", "math", "logic", "code", "science"]

NET_SPECS = [
    ("N1", "WorldKnowledge", "general", "general"),
    ("N2", "Math", "math", "math"),
    ("N3", "Logic", "logic", "logic"),
    ("N4", "Language", "code", "code"),
    ("N5", "Verification", "science", "science"),
]


def make_phase5_stream(config: DSCNSConfig, data: Dict[str, Any],
                       rng: np.random.RandomState) -> List[List[Dict[str, Any]]]:
    """general(5) -> code(5) -> mixed_code(5) -> science(5) = 20 rounds.

    Sampling uses replacement because the HumanEval code pool is small;
    P5 focuses on the self-modification closed loop, not stream curation.
    """
    train = data["train"]
    per_round = config.samples_per_round
    phases = [("general", 5), ("code", 5), ("mixed_code", 5), ("science", 5)]
    stream = []
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
                                                    replace=True)]
                rng.shuffle(samples)
                stream.append(samples[:per_round])
            else:
                pool = train[phase]
                k = min(per_round, len(pool))
                stream.append([{"text": t, "domain": phase, "source": phase,
                                "reliability": 0.8}
                               for t in rng.choice(pool, size=k, replace=True)])
    # respect config.num_rounds (used by --rounds for quick trials)
    n_rounds = int(getattr(config, "num_rounds", 20))
    return stream[:n_rounds]


def build_phase5_networks(base: Any, config: DSCNSConfig) -> List[CognitiveNetwork]:
    """Create num_networks CognitiveNetworks with optional plasticity modules."""
    n = int(getattr(config, "num_networks", 5))
    networks = []
    for net_id, name, domain, data_domain in NET_SPECS[:n]:
        base.add_adapter(net_id)
        plasticity = None
        if getattr(config, "enable_plasticity", False):
            plasticity = IntrinsicPlasticityModule(
                hidden_dim=getattr(config, "plasticity_hidden_dim", 768),
                adapter_dim=config.lora_r,
                meta_dim=getattr(config, "meta_dim", 32),
                plasticity_rank=getattr(config, "plasticity_rank", 8),
                use_hidden=getattr(config, "use_hidden", True),
                use_param_stats=getattr(config, "use_param_stats", True),
                use_meta=getattr(config, "use_meta", True),
                modulation_strength_init=getattr(
                    config, "modulation_strength_init", 0.05),
            ).to(base.peft_model.device)
        net = CognitiveNetwork(
            net_id=net_id, name=name, domain=domain,
            peft_model=base.peft_model, memory=MemorySystem(),
            domain_embedding=None,
            base_lr=getattr(config, "task_lr", 5e-5),
            plasticity=plasticity,
            plasticity_cfg={"meta_dim": getattr(config, "meta_dim", 32)},
        )
        net.data_domain = data_domain
        net.set_trainable(False)
        networks.append(net)
    return networks


def train_round_step(net: CognitiveNetwork, base: Any, texts: List[str],
                     config: DSCNSConfig, batch_size: int = 8,
                     max_len: int = 192) -> float:
    """Standard task learning on the round's texts (external to plasticity).

    Mirrors the Control-mode training helper; increments the network's
    step counter so the external trigger can fire on a fixed cadence.
    """
    import torch

    net.set_trainable(True)
    base.peft_model.set_adapter(net.id)
    base.peft_model.train()
    opt = net.get_optimizer()
    for g in opt.param_groups:
        g["lr"] = getattr(config, "task_lr", 5e-5)
    opt.zero_grad()
    enc = base.tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len)
    enc = {k: v.to(base.device) for k, v in enc.items()}
    n_chunks = max(1, (len(texts) + batch_size - 1) // batch_size)
    loss_sum = 0.0
    for i in range(0, len(texts), batch_size):
        chunk = {k: v[i:i + batch_size] for k, v in enc.items()}
        out = base.peft_model(**chunk, labels=chunk["input_ids"])
        (out.loss / n_chunks).backward()
        loss_sum += float(out.loss.item())
        del out, chunk
        torch.nn.utils.clip_grad_norm_(
            [p for p in net._adapter_params() if p.grad is not None], max_norm=1.0)
        opt.step()
        opt.zero_grad()
        net.step_count += 1
    base.peft_model.eval()
    net.set_trainable(False)
    return loss_sum / n_chunks


def quick_validation(net: CognitiveNetwork, base: Any, texts: List[str],
                     config: DSCNSConfig, before_loss: Optional[float] = None
                     ) -> tuple:
    """Experiment-controller safety check: the modification must not explode.

    Accepts when the CE loss did not rise by more than
    ``validation_loss_margin`` nats (relative criterion; GPT-2 raw perplexity
    varies across domains, so an absolute perplexity cap alone is unreliable).
    """
    loss = float(np.mean(net.losses_for_texts(
        texts[: getattr(config, "quick_validation_samples", 8)],
        base.tokenizer, batch_size=8, max_len=config.max_len)))
    ppl = float(np.exp(loss))
    ok, reason = True, "ok"
    if before_loss is not None and loss > before_loss + \
            getattr(config, "validation_loss_margin", 0.5):
        ok, reason = False, "loss_exploded"
    cap = getattr(config, "validation_perplexity_cap", 100.0)
    if ppl > cap and (before_loss is None or
                      ppl > float(np.exp(before_loss)) * 2.0):
        ok, reason = False, "ppl_cap"
    return ok, reason, float(loss), float(ppl)


def make_random_delta(reference_delta: Dict[str, Any],
                      device: Any, rng: Optional[np.random.RandomState] = None
                      ) -> Dict[str, Any]:
    """Random delta with the same total norm as the reference (Control A)."""
    import torch

    dA = torch.randn_like(reference_delta["delta_W_A"])
    dB = torch.randn_like(reference_delta["delta_W_B"])
    ref_norm = (reference_delta["delta_W_A"].norm() +
                reference_delta["delta_W_B"].norm())
    rand_norm = dA.norm() + dB.norm()
    scale = float(ref_norm) / float(max(rand_norm, 1e-12))
    return {
        "delta_W_A": dA * scale,
        "delta_W_B": dB * scale,
        "modulation_strength": reference_delta.get("modulation_strength", 0.0),
    }


def apply_plasticity_step(net: CognitiveNetwork, base: Any,
                          batch: List[Dict[str, Any]],
                          config: DSCNSConfig,
                          mode: str = "intrinsic",
                          rng: Optional[np.random.RandomState] = None,
                          constant_delta: Optional[Dict[str, Any]] = None,
                          other_texts: Optional[List[str]] = None
                          ) -> Dict[str, Any]:
    """One plasticity step through the accept/rollback protocol.

    mode: "intrinsic" | "random" | "constant" | "shuffled"
    """
    texts = [e["text"] for e in batch]
    params_before = net.snapshot_parameters()
    rec: Dict[str, Any] = {"mode": mode, "applied": False, "accepted": False}

    with net._no_grad_ctx():
        if mode == "intrinsic":
            delta = net.generate_delta(texts, base.tokenizer,
                                       max_len=config.max_len)
        elif mode == "random":
            # reference norm comes from a precomputed intrinsic delta
            ref = getattr(net, "_random_reference_delta", None)
            if ref is None:
                delta = net.generate_delta(texts, base.tokenizer,
                                           max_len=config.max_len)
                net._random_reference_delta = delta
                ref = delta
            delta = make_random_delta(ref, base.device, rng)
        elif mode == "constant":
            if constant_delta is None:
                cached = getattr(net, "_constant_delta", None)
                if cached is None:
                    cached = net.generate_delta(texts, base.tokenizer,
                                                max_len=config.max_len)
                    net._constant_delta = cached
                delta = cached
        elif mode == "shuffled":
            other = other_texts or texts
            delta = net.generate_delta(other, base.tokenizer,
                                       max_len=config.max_len)
        else:
            raise ValueError(f"unknown delta mode: {mode}")

    delta_norm = float(delta["delta_W_A"].norm()) + \
        float(delta["delta_W_B"].norm())
    rec.update({"delta_norm": delta_norm,
                "modulation_strength": float(delta.get("modulation_strength", 0.0))})
    if delta_norm < getattr(config, "min_delta_threshold", 1e-6):
        rec["reason"] = "delta_too_small"
        return rec

    before_loss = float(np.mean(net.losses_for_texts(
        texts[: getattr(config, "quick_validation_samples", 8)],
        base.tokenizer, batch_size=8, max_len=config.max_len)))

    # behavioral-change evidence (Test 4 metric): argmax prediction change
    import torch

    sub = texts[: getattr(config, "quick_validation_samples", 8)]
    enc = base.tokenizer(sub, return_tensors="pt", padding=True,
                         truncation=True, max_length=config.max_len)
    enc = {k: v.to(base.device) for k, v in enc.items()}
    net.peft_model.set_adapter(net.id)
    net.peft_model.eval()
    with torch.no_grad():
        logits_before = net.peft_model(**enc).logits

    net.apply_intrinsic_modification(delta, alpha=config.plasticity_alpha)
    rec["applied"] = True

    with torch.no_grad():
        logits_after = net.peft_model(**enc).logits
    pred_change = float((logits_before.argmax(-1) !=
                         logits_after.argmax(-1)).float().mean())
    logits_diff = float((logits_after - logits_before).abs().mean())
    rec.update({"pred_change_rate": pred_change, "logits_diff": logits_diff})

    ok, reason, loss_after, ppl = quick_validation(
        net, base, texts, config, before_loss=before_loss)
    rec.update({"loss_before": before_loss, "loss_after": loss_after,
                "perplexity": ppl, "reason": reason})
    if ok:
        rec["accepted"] = True
        if mode == "shuffled":
            # restore: shuffled pairing is only an analysis, not a kept change
            net.restore_parameters(params_before)
    else:
        net.restore_parameters(params_before)
    return rec


def eval_matched_per_domain(networks: List[CognitiveNetwork],
                            eval_sets: Dict[str, List[str]], base: Any,
                            n: int = 48) -> Dict[str, float]:
    """Perf (exp(-loss)) of each domain-matched network on its own eval set."""
    out = {}
    by_domain = {net.data_domain: net for net in networks
                 if net.data_domain is not None}
    for domain, texts in eval_sets.items():
        net = by_domain.get(domain) or networks[0]
        out[domain] = net.evaluate_texts(texts[:n], base.tokenizer,
                                         batch_size=8)
    return out


def update_meta_from_probe(networks: List[CognitiveNetwork],
                           probe_sets: Dict[str, List[str]], base: Any,
                           domain: str, n: int = 16) -> None:
    """Refresh per-network competence on the current phase's probe texts."""
    for net in networks:
        pool = probe_sets.get(domain) or probe_sets.get("general") or []
        if not pool:
            continue
        p = net.evaluate_texts(pool[:n], base.tokenizer, batch_size=8)
        net.performance_history.append(p)
        net.competence = float(p)
        net.uncertainty = float(np.clip(1.0 - p, 0.0, 1.0))
