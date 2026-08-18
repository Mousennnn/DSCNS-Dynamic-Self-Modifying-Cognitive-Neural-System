"""P5 long-horizon experiment v2 (design report v2).

Two lines:

  (A) 150-round strict validation, four groups sharing one frozen Probe Set:
        p5_150             -- real P5 intrinsic self-modification
        random_control_150 -- modification budget matched (same count, same
                              per-round magnitude), directions random
        no_modification_150-- operational loop runs, theta stays frozen
        baseline           -- initial-state evaluation only (t=0 reference)

  (B) p5_3000 -- extreme long-horizon run of the real P5 loop (3000 rounds,
      probe every 10 rounds, checkpoint every 100 rounds, no early stop).

The frozen Phase-5 implementation is used unchanged (commit 3208463, v0.3.0):
this script only orchestrates the existing model-side loop and logs metrics.
No P5 core file is modified; no tuning; no new controller; the external
trigger is the fixed every-round schedule of the existing P5 loop.

New metrics vs the previous 150-run:
  gross movement   D_gross(T) = sum_t ||theta_t - theta_{t-1}||
  true net drift   D_net(T)   = ||theta_T - theta_0||
  drift ratio      R = D_net / (D_gross + eps)
  layer-wise drift per transformer layer (drift + relative drift)
  SHA-256 parameter hash (proves actual tensor change; constant for No-Mod)
  checkpoint state separation: model_state / optimizer_state (null: no
  optimizer in this loop) / p5_state / step / seed / probe_output / hash
  probe output drift on the frozen Probe Set (vs t=0 and vs previous eval)
  consecutive-delta direction cosine (P5 vs Random-Control)

Run:  python scripts/run_p5_long_horizon.py [--smoke]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import make_base_model, make_config, prepare_data
from dscns.utils import set_seed
from phase5_common import build_phase5_networks

BASE_DIR = os.path.join("experiments", "p5_long_horizon")
SEED = 42
PROBE_SEED = 1234          # probe set is frozen with its own deterministic seed
PROBE_SIZE = 32            # 32 probe texts (4 batches of 8)
BATCH_SIZE = 8             # operational-loop batch (texts per round)
MAX_LEN = 192

PROBE_DOMAIN_COUNTS = {"general": 8, "math": 6, "logic": 6, "code": 6,
                       "science": 6}  # sums to 32


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def entropy_of(logits) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).sum(-1).mean())


def theta_norm(net) -> float:
    t = net._current_params_tensors()
    return float(t["W_A"].norm()) + float(t["W_B"].norm())


def param_hash(net) -> str:
    """SHA-256 over the network's adapter parameters (sorted by name)."""
    h = hashlib.sha256()
    names = []
    for n, p in net.peft_model.named_parameters():
        if f".{net.id}." in n and "lora" in n:
            names.append(n)
    for n in sorted(names):
        p = dict(net.peft_model.named_parameters())[n]
        h.update(n.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def net_drift(net, theta0) -> float:
    drift = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in theta0.get("lora_A", {}):
                drift += float((p.data - theta0["lora_A"][n]).norm())
            elif n in theta0.get("lora_B", {}):
                drift += float((p.data - theta0["lora_B"][n]).norm())
    return drift


def applied_change(net, before) -> float:
    change = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in before.get("lora_A", {}):
                change += float((p.data - before["lora_A"][n]).norm())
            elif n in before.get("lora_B", {}):
                change += float((p.data - before["lora_B"][n]).norm())
    return change


def layer_wise(net, theta0) -> dict:
    """{layer_idx: {"drift": D_l, "base_norm": ||theta_l(0)||, "rel": R_l}}."""
    drift = {}
    base = {}
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            m = re.search(r"transformer\.h\.(\d+)", n)
            if not m or "lora" not in n:
                continue
            i = int(m.group(1))
            if n in theta0.get("lora_A", {}):
                b = theta0["lora_A"][n]
            elif n in theta0.get("lora_B", {}):
                b = theta0["lora_B"][n]
            else:
                continue
            drift[i] = drift.get(i, 0.0) + float((p.data - b).norm())
            base[i] = base.get(i, 0.0) + float(b.norm())
    return {i: {"drift": drift.get(i, 0.0),
                "base_norm": base.get(i, 0.0),
                "rel": drift.get(i, 0.0) / max(base.get(i, 0.0), 1e-12)}
            for i in sorted(drift.keys())}


def make_probe_set(data) -> list:
    """Frozen probe set: PROBE_SIZE texts, deterministic (PROBE_SEED)."""
    rng = np.random.RandomState(PROBE_SEED)
    probes = []
    for domain, k in PROBE_DOMAIN_COUNTS.items():
        pool = data["train"][domain]
        chosen = [str(t) for t in
                  rng.choice(pool, size=min(k, len(pool)), replace=False)]
        for t in chosen:
            probes.append({"id": len(probes) + 1, "domain": domain, "text": t})
    return probes[:PROBE_SIZE]


def make_operational_stream(data, num_rounds):
    """20-round phase pattern repeated; sampling with replacement."""
    rng = np.random.RandomState(SEED)
    train = data["train"]
    pattern = [("general", 5), ("code", 5), ("mixed_code", 5), ("science", 5)]
    rounds = []
    while len(rounds) < num_rounds:
        for phase, n in pattern:
            for _ in range(n):
                if len(rounds) >= num_rounds:
                    break
                if phase == "mixed_code":
                    samples = []
                    for dom, frac in [("code", 0.5), ("general", 0.25),
                                      ("science", 0.25)]:
                        pool = train[dom]
                        k = max(1, int(BATCH_SIZE * frac))
                        samples += [str(t) for t in
                                    rng.choice(pool, size=min(k, len(pool)),
                                               replace=True)]
                    rng.shuffle(samples)
                    rounds.append(samples[:BATCH_SIZE])
                else:
                    pool = train[phase]
                    k = min(BATCH_SIZE, len(pool))
                    rounds.append([str(t) for t in
                                   rng.choice(pool, size=k, replace=True)])
    return rounds[:num_rounds]


def random_delta(target_norm: float, device, strength=0.05, gen=None) -> dict:
    """Budget-matched random direction: same shape, same total norm as P5."""
    kw = {} if gen is None else {"generator": gen}
    eA = torch.randn(768, 16, device=device, **kw)
    eB = torch.randn(16, 768, device=device, **kw)
    n = eA.norm() + eB.norm()
    s = float(target_norm) / float(max(n, 1e-12))
    return {"delta_W_A": eA * s, "delta_W_B": eB * s,
            "modulation_strength": strength}


def probe_eval(net, base, probes, max_len, ref_batches=None,
               prev_batches=None):
    """Evaluate the frozen probe set batch-wise (memory-safe).

    Returns (out_d, batches, prev) where:
      out_d      -- probe_loss, probe_entropy, output_drift_vs_0 (vs the
                    reference evaluation) and output_drift_vs_prev;
      batches    -- list of fp32 logits batches (the reference snapshot);
      prev       -- list of fp16 logits batches (for the next comparison).

    Accumulates all aggregates per batch so the full probe logits tensor is
    never materialized at once (fits the 8 GB GPU).  log-softmax is computed
    once and reused for both the entropy and the CE loss, and the CUDA
    caching allocator is compacted after each call (long-horizon runs
    otherwise degrade 20-30x from fragmentation of the large logits blocks).
    """
    import torch
    import torch.nn.functional as F

    texts = [p["text"] for p in probes]
    net.peft_model.set_adapter(net.id)
    net.peft_model.eval()
    pad = base.tokenizer.pad_token_id or -100
    loss_sum = 0.0
    n_tok = 0
    ent_sum = 0.0
    n_el = 0
    d0_sum = 0.0
    dp_sum = 0.0
    n_b = 0
    batches, prev = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            chunk = texts[i:i + 8]
            enc = base.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=max_len)
            enc = {k: v.to(net.peft_model.device) for k, v in enc.items()}
            out = net.peft_model(**enc, labels=enc["input_ids"])
            lg = out.logits.detach()
            batches.append(lg)
            prev.append(lg.half())
            logp = torch.log_softmax(lg.float().view(-1, lg.size(-1)), dim=-1)
            logp = logp.view(lg.size(0), lg.size(1), lg.size(-1))
            ent_sum += float(-(torch.exp(logp) * logp).sum(-1).mean()) * \
                (lg.shape[0] * lg.shape[1])
            n_el += lg.shape[0] * lg.shape[1]
            lab = enc["input_ids"][:, 1:].reshape(-1)
            ce = F.nll_loss(logp[:, :-1].reshape(-1, logp.size(-1)), lab,
                            reduction="sum", ignore_index=pad)
            loss_sum += float(ce)
            n_tok += int((lab != pad).sum())
            if ref_batches is not None:
                d0_sum += float((lg - ref_batches[n_b]).abs().mean())
            if prev_batches is not None:
                dp_sum += float((lg - prev_batches[n_b].float()).abs().mean())
            n_b += 1
    out_d = {"probe_loss": loss_sum / max(n_tok, 1),
             "probe_entropy": ent_sum / max(n_el, 1)}
    if ref_batches is not None:
        out_d["output_drift_vs_0"] = d0_sum / max(n_b, 1)
    if prev_batches is not None:
        out_d["output_drift_vs_prev"] = dp_sum / max(n_b, 1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_d, batches, prev


def freeze_config_yaml(path, group_cfg: dict):
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(group_cfg, f, allow_unicode=True, sort_keys=False)


def run_group(group: str, mode: str, num_rounds: int, config, data, probes,
              probe_every: int, ckpt_every: int, smoke: bool,
              random_norms=None, group_idx: int = 0) -> dict:
    """Run one experimental group on a fresh, identically-initialized network."""
    t0 = time.time()
    set_seed(SEED)                      # identical theta_0 across groups
    base = make_base_model(config, tag=f"p5l_{group}")
    net = build_phase5_networks(base, config)[0]
    tokenizer = base.tokenizer
    # dedicated RNG for random-control delta draws (isolated from init)
    delta_gen = torch.Generator(device=base.device)
    delta_gen.manual_seed(SEED + 1000 + group_idx)
    stream = make_operational_stream(data, num_rounds)
    theta0 = net.snapshot_parameters()
    hash0 = param_hash(net)
    norm0 = theta_norm(net)

    out_dir = os.path.join(BASE_DIR, "results", group)
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", group)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # reference probe evaluation at t=0 (also serves the Baseline group)
    probe_meta, logits0, prev0 = probe_eval(net, base, probes, MAX_LEN)
    prev_snap = prev0
    probe_meta0 = {"round": 0, "probe_loss": probe_meta["probe_loss"],
                   "probe_entropy": probe_meta["probe_entropy"],
                   "output_drift_vs_0": 0.0, "output_drift_vs_prev": 0.0}
    rows = [probe_meta0]
    gross = 0.0
    prev_flat = None
    prev_hash = hash0
    layer_curve = {}
    drift_curve = []
    first_nan = first_inf = None

    for r, texts in enumerate(stream):
        if smoke and r >= num_rounds:
            break
        round_no = r + 1
        row = {"round": round_no}

        # ---- modification step (frozen P5 mechanism or controls) ----
        delta_norm = 0.0
        if mode == "p5":
            delta = net.generate_delta(texts, tokenizer, max_len=MAX_LEN)
            delta_norm = float(delta["delta_W_A"].norm()) + \
                float(delta["delta_W_B"].norm())
            d_nan, d_inf = nan_inf(delta["delta_W_A"], delta["delta_W_B"])
            before = net.snapshot_parameters()
            net.apply_intrinsic_modification(delta, alpha=config.plasticity_alpha)
            applied = applied_change(net, before)
            flat = torch.cat([delta["delta_W_A"].flatten(),
                              delta["delta_W_B"].flatten()])
        elif mode == "random":
            target = float(random_norms[round_no - 1])
            delta = random_delta(target, net.peft_model.device, gen=delta_gen)
            delta_norm = float(delta["delta_W_A"].norm()) + \
                float(delta["delta_W_B"].norm())
            d_nan, d_inf = nan_inf(delta["delta_W_A"], delta["delta_W_B"])
            before = net.snapshot_parameters()
            net.apply_intrinsic_modification(delta, alpha=config.plasticity_alpha)
            applied = applied_change(net, before)
            flat = torch.cat([delta["delta_W_A"].flatten(),
                              delta["delta_W_B"].flatten()])
        else:  # no_modification: loop runs, theta stays frozen
            delta_norm, applied, d_nan, d_inf, flat = 0.0, 0.0, 0, 0, None

        # ---- drift accounting ----
        gross += applied
        dnet = net_drift(net, theta0)
        drift_curve.append(dnet)
        row.update({
            "delta_norm": delta_norm,
            "theta_norm": theta_norm(net),
            "applied_change": applied,
            "gross_drift": gross,
            "net_drift": dnet,
            "drift_ratio": dnet / (gross + 1e-12),
            "nan_count": d_nan, "inf_count": d_inf,
        })
        if d_nan > 0 and first_nan is None:
            first_nan = round_no
        if d_inf > 0 and first_inf is None:
            first_inf = round_no

        # direction statistics (consecutive deltas)
        if flat is not None and prev_flat is not None:
            row["delta_cosine"] = float(
                (flat * prev_flat).sum() /
                max(flat.norm() * prev_flat.norm(), 1e-12))
        prev_flat = flat

        # parameter hash proof (per round for No-Mod, at checkpoints otherwise)
        if mode == "no_modification" or (ckpt_every and round_no % ckpt_every == 0):
            h = param_hash(net)
            row["param_hash"] = h
            row["hash_changed"] = (h != prev_hash)
            prev_hash = h

        # layer-wise drift
        lw = layer_wise(net, theta0)
        layer_curve[round_no] = lw
        row["layer_drift_sum"] = sum(v["drift"] for v in lw.values())

        # probe evaluation (fixed schedule)
        if round_no % probe_every == 0 or smoke:
            pm, _, prev_snap = probe_eval(net, base, probes, MAX_LEN,
                                          ref_batches=logits0,
                                          prev_batches=prev_snap)
            row["probe_loss"] = pm["probe_loss"]
            row["probe_entropy"] = pm["probe_entropy"]
            row["output_drift_vs_0"] = pm["output_drift_vs_0"]
            row["output_drift_vs_prev"] = pm["output_drift_vs_prev"]
        rows.append(row)

        if smoke:
            if round_no >= num_rounds:
                break
        if (not smoke) and (ckpt_every and round_no % ckpt_every == 0):
            save_checkpoint(ckpt_dir, round_no, net, theta0, row, hash0,
                            SEED, group)

    # final probe + final checkpoint
    pmf, _, _ = probe_eval(net, base, probes, MAX_LEN, ref_batches=logits0)
    final_row = {
        "round": num_rounds + 1, "delta_norm": delta_norm,
        "theta_norm": theta_norm(net), "applied_change": applied,
        "gross_drift": gross, "net_drift": net_drift(net, theta0),
        "drift_ratio": net_drift(net, theta0) / (gross + 1e-12),
        "nan_count": d_nan, "inf_count": d_inf,
        "probe_loss": pmf["probe_loss"], "probe_entropy": pmf["probe_entropy"],
        "output_drift_vs_0": pmf["output_drift_vs_0"],
        "output_drift_vs_prev": pmf.get("output_drift_vs_prev", 0.0),
    }
    rows.append(final_row)
    final_hash = param_hash(net)
    if not smoke:
        save_checkpoint(ckpt_dir, num_rounds + 1, net, theta0, final_row,
                        hash0, SEED, group)

    # ---- persistence ----
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "group": group, "mode": mode, "num_rounds": num_rounds,
        "seed": SEED, "git_commit": git_sha(),
        "hash0": hash0, "hash_final": final_hash,
        "hash_constant": final_hash == hash0,
        "param_norm_0": norm0, "param_norm_final": theta_norm(net),
        "gross_movement": gross, "true_net_drift": net_drift(net, theta0),
        "drift_ratio": net_drift(net, theta0) / (gross + 1e-12),
        "max_single_step_drift": float(max((r["applied_change"]
                                            for r in rows if r["round"] > 0),
                                           default=0.0)),
        "mean_single_step_drift": float(np.mean(
            [r["applied_change"] for r in rows if r["round"] > 0] or [0.0])),
        "first_nan_round": first_nan, "first_inf_round": first_inf,
        "probe_output_drift_final": pmf["output_drift_vs_0"],
        "probe_output_drift_mean": float(np.mean(
            [r.get("output_drift_vs_0", 0.0) for r in rows])),
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "group.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "layer_drift.json"), "w",
              encoding="utf-8") as f:
        json.dump(layer_curve, f, indent=1)
    print(f"[{group}] rounds={num_rounds} mode={mode} "
          f"hash0==final: {final_hash == hash0} "
          f"net_drift={summary['true_net_drift']:.4f} "
          f"gross={gross:.4f} ratio={summary['drift_ratio']:.4f} "
          f"param_norm {norm0:.3f}->{theta_norm(net):.3f} "
          f"probe_drift={pmf['output_drift_vs_0']:.6f} "
          f"nan={first_nan} inf={first_inf} ({time.time() - t0:.0f}s)", flush=True)
    return summary


def save_checkpoint(ckpt_dir, step, net, theta0, row, hash0, seed, group):
    """Checkpoint with strict state separation (.pt; gitignored).

    model_state / optimizer_state / p5_state are kept distinct; the loop
    has no optimizer or training state, so optimizer_state is null by
    construction (drift can only come from intrinsic modification).
    """
    model_state = {}
    for n, p in net.peft_model.named_parameters():
        if f".{net.id}." in n and "lora" in n:
            model_state[n] = p.detach().cpu().clone()
    p5_state = {"plasticity": net.plasticity.state_dict() if net.plasticity
                else None,
                "num_modifications": net.plasticity.num_modifications
                if net.plasticity else 0,
                "step_count": net.step_count}
    ck = {
        "step": step, "seed": seed, "group": group,
        "model_state": model_state,
        "optimizer_state": None,          # no optimizer in this loop
        "p5_state": p5_state,
        "parameter_hash": param_hash(net),
        "hash0": hash0,
        "probe_output": {k: row.get(k) for k in
                         ["output_drift_vs_0", "output_drift_vs_prev",
                          "probe_loss", "probe_entropy"]},
        "theta_norm": row.get("theta_norm"),
        "net_drift": row.get("net_drift"),
        "gross_drift": row.get("gross_drift"),
    }
    torch.save(ck, os.path.join(ckpt_dir, f"checkpoint_{step:06d}.pt"))


def nan_inf(*tensors):
    nan_c = inf_c = 0
    for t in tensors:
        nan_c += int(torch.isnan(t).sum())
        inf_c += int(torch.isinf(t).sum())
    return nan_c, inf_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="quick validation run (6 rounds per group, no 3000)")
    ap.add_argument("--only-3000", action="store_true",
                    help="run only the p5_3000 group (150-round results already saved)")
    args = ap.parse_args()
    smoke = args.smoke
    only_3000 = args.only_3000
    rounds_150 = 6 if smoke else 150
    probe_every_150 = 1
    ckpt_every_150 = 50
    rounds_3000 = 12 if smoke else 3000
    probe_every_3000 = 6 if smoke else 10
    ckpt_every_3000 = 6 if smoke else 100

    os.makedirs(BASE_DIR, exist_ok=True)
    for d in ["config", "probe_set", "results", "checkpoints", "plots"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

    config = make_config(cfg_path="config/phase5.yaml")
    config.num_networks = 1
    config.seed = SEED
    set_seed(SEED)
    data = prepare_data(config)

    # ---- freeze the probe set once (shared by every group) ----
    probes = make_probe_set(data)
    with open(os.path.join(BASE_DIR, "probe_set", "probes.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": PROBE_SEED, "count": len(probes),
                   "domains": PROBE_DOMAIN_COUNTS, "probes": probes},
                  f, indent=1, ensure_ascii=False)
    for p in probes:
        with open(os.path.join(BASE_DIR, "probe_set",
                               f"probe_{p['id']:04d}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(p["text"])

    # ---- freeze config yamls ----
    common_cfg = {
        "git_commit": git_sha(), "seed": SEED,
        "model": {"name": config.model_name, "lora_r": config.lora_r,
                  "lora_alpha": config.lora_alpha,
                  "lora_dropout": config.lora_dropout},
        "plasticity": {"alpha": config.plasticity_alpha,
                       "rank": config.plasticity_rank,
                       "meta_dim": config.meta_dim,
                       "modulation_strength_init":
                           config.modulation_strength_init},
        "data": {"batch_size": BATCH_SIZE, "max_len": MAX_LEN,
                 "probe_set": {"seed": PROBE_SEED, "count": PROBE_SIZE}},
        "loop": {"network": "N1", "apply_every_round": True,
                 "validation_rollback": False},
    }
    groups_yaml = [
        ("p5_150", {"group": "p5_150", "mode": "p5", "rounds": rounds_150,
                    "probe_every": probe_every_150,
                    "checkpoint_every": ckpt_every_150}),
        ("random_control_150", {"group": "random_control_150",
                                "mode": "random", "rounds": rounds_150,
                                "probe_every": probe_every_150,
                                "checkpoint_every": ckpt_every_150,
                                "note": "budget matched to p5_150: same "
                                        "per-round ||delta|| magnitudes, "
                                        "random directions"}),
        ("no_modification_150", {"group": "no_modification_150",
                                 "mode": "no_modification",
                                 "rounds": rounds_150,
                                 "probe_every": probe_every_150,
                                 "checkpoint_every": ckpt_every_150}),
        ("p5_3000", {"group": "p5_3000", "mode": "p5", "rounds": rounds_3000,
                     "probe_every": probe_every_3000,
                     "checkpoint_every": ckpt_every_3000,
                     "note": "extreme long-horizon run; no early stop"})]
    for name, gc in groups_yaml:
        freeze_config_yaml(os.path.join(BASE_DIR, "config", f"{name}.yaml"),
                           {**common_cfg, **gc})

    # ---- execution order (design report section 22) ----
    summaries = {}
    if only_3000:
        # 150-round validation already saved; keep their summaries.
        sp = os.path.join(BASE_DIR, "results", "summaries.json")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                summaries.update(json.load(f))
    else:
        # 1) P5-OM 150 (also provides the budget profile for Random-Control)
        s = run_group("p5_150", "p5", rounds_150, config, data, probes,
                      probe_every_150, ckpt_every_150, smoke, group_idx=0)
        summaries["p5_150"] = s

        # 2) Random-Control 150 with matched magnitudes from p5_150
        with open(os.path.join(BASE_DIR, "results", "p5_150", "metrics.json"),
                  encoding="utf-8") as f:
            p5_rows = json.load(f)
        norms = [r["delta_norm"] for r in p5_rows if r["round"] > 0
                 and r["round"] <= rounds_150]
        if smoke:
            norms = norms[:rounds_150]
        s = run_group("random_control_150", "random", rounds_150, config, data,
                      probes, probe_every_150, ckpt_every_150, smoke,
                      random_norms=norms, group_idx=1)
        summaries["random_control_150"] = s

        # 3) No-Modification 150
        s = run_group("no_modification_150", "no_modification", rounds_150,
                      config, data, probes, probe_every_150, ckpt_every_150,
                      smoke, group_idx=2)
        summaries["no_modification_150"] = s

        # 4) Baseline (t=0 evaluation only; reference outputs for all groups)
        baseline_summary = run_baseline(config, data, probes)
        summaries["baseline"] = baseline_summary

    # 5) P5-OM 3000 (extreme run; only after validation confirms logging)
    if not smoke:
        s = run_group("p5_3000", "p5", rounds_3000, config, data, probes,
                      probe_every_3000, ckpt_every_3000, smoke, group_idx=3)
        summaries["p5_3000"] = s

    with open(os.path.join(BASE_DIR, "results", "summaries.json"), "w",
              encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print("=== long-horizon v2 done ===")
    for g, s2 in summaries.items():
        print(f"  {g}: net={s2.get('true_net_drift', 0):.4f} "
              f"gross={s2.get('gross_movement', 0):.4f} "
              f"ratio={s2.get('drift_ratio', 0):.4f} "
              f"hash_const={s2.get('hash_constant')} "
              f"probe={s2.get('probe_output_drift_final', 0):.6f}")


def run_baseline(config, data, probes):
    """Baseline: no loop, no modification — initial-state probe evaluation."""
    set_seed(SEED)
    base = make_base_model(config, tag="p5l_baseline")
    net = build_phase5_networks(base, config)[0]
    pm, _, _ = probe_eval(net, base, probes, MAX_LEN)
    out_dir = os.path.join(BASE_DIR, "results", "baseline")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "group": "baseline", "mode": "baseline", "num_rounds": 0,
        "seed": SEED, "git_commit": git_sha(),
        "hash0": param_hash(net),
        "param_norm_0": theta_norm(net),
        "probe_loss": pm["probe_loss"],
        "probe_entropy": pm["probe_entropy"],
        "note": "initial state reference (theta_0); no modification",
    }
    with open(os.path.join(out_dir, "group.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[baseline] hash0={summary['hash0'][:16]}... "
          f"norm0={summary['param_norm_0']:.3f} "
          f"probe_loss={pm['probe_loss']:.4f} "
          f"probe_entropy={pm['probe_entropy']:.4f}", flush=True)
    return summary


if __name__ == "__main__":
    main()
