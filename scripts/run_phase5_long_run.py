"""P5 150-round long-horizon intrinsic self-modification stress test.

This is a LONG-HORIZON / STRESS-TEST experiment on the **frozen** Phase 5
implementation (commit 3208463, v0.3.0).  It does NOT modify any P5 core
mechanism: it only orchestrates the existing model-side loop

    theta_t -> h_t -> delta_theta_t = P_phi(h_t, stats(theta_t), s_t)
            -> theta_{t+1} = theta_t + alpha * delta_theta_t -> ...

for 150 continuous rounds on a single cognitive network (N1) without any
re-initialization, accept/rollback, or controller.  This is the strongest
stability test of the intrinsic mechanism itself: every round's delta is
applied unconditionally (validation/rollback are experiment-controller
concerns in the P5 design report section 8 and are intentionally disabled
here to expose the raw closed-loop dynamics).

Outputs (experiments/phase5_long_run/):
    config.json  metrics.json  metrics.csv  summary.json
    modification_norm_vs_round.png  relative_modification_vs_round.png
    parameter_norm_vs_round.png     cumulative_parameter_drift.png
    behavioral_drift_vs_round.png   entropy_vs_round.png
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from common import make_base_model, make_config, prepare_data
from dscns.utils import set_seed
from phase5_common import build_phase5_networks

OUT_DIR = os.path.join("experiments", "phase5_long_run")
NUM_ROUNDS = 150
BATCH_SIZE = 8          # texts per round (stress-loop batch; recorded in config)
SEED = 42


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_long_stream(config, data, rng, num_rounds=150):
    """150 rounds from the frozen 20-round phase pattern (with replacement).

    Pattern general(5) -> code(5) -> mixed_code(5) -> science(5) repeated
    7.5 times; sampling with replacement (small HumanEval pool).  This is
    stream construction for the stress loop, not a P5 mechanism change.
    """
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


def _entropy(logits) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = torch.exp(logp)
    ent = -(p * logp).sum(-1)
    return float(ent.mean())


def _theta_norm(net) -> float:
    t = net._current_params_tensors()
    return float(t["W_A"].norm()) + float(t["W_B"].norm())


def _param_drift(net, theta0) -> float:
    """||theta_t - theta_0||_2 over the network's adapter parameters."""
    drift = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in theta0.get("lora_A", {}):
                drift += float((p.data - theta0["lora_A"][n]).norm())
            elif n in theta0.get("lora_B", {}):
                drift += float((p.data - theta0["lora_B"][n]).norm())
    return drift


def _applied_change(net, before) -> float:
    """||theta_{t+1} - theta_t||_2 actually applied this round."""
    change = 0.0
    with torch.no_grad():
        for n, p in net.peft_model.named_parameters():
            if n in before.get("lora_A", {}):
                change += float((p.data - before["lora_A"][n]).norm())
            elif n in before.get("lora_B", {}):
                change += float((p.data - before["lora_B"][n]).norm())
    return change


def _nan_inf_counts(*tensors) -> tuple:
    nan_c, inf_c = 0, 0
    for t in tensors:
        nan_c += int(torch.isnan(t).sum())
        inf_c += int(torch.isinf(t).sum())
    return nan_c, inf_c


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- frozen config source: the existing P5 config, seed 42 ----
    config = make_config(cfg_path="config/phase5.yaml")
    config.num_networks = 1          # single theta-chain (recorded; yaml untouched)
    config.seed = SEED
    set_seed(config.seed)
    data = prepare_data(config)
    rng = np.random.RandomState(SEED)
    stream = make_long_stream(config, data, rng, NUM_ROUNDS)

    base = make_base_model(config, tag="p5_long")
    networks = build_phase5_networks(base, config)
    net = networks[0]
    tokenizer = base.tokenizer
    max_len = config.max_len
    alpha = config.plasticity_alpha

    theta0 = net.snapshot_parameters()
    metrics = []

    for r, texts in enumerate(stream):
        # ---- state before this round's modification (theta_t) ----
        logits_before = net._logits_for_texts(texts, tokenizer, max_len=max_len)
        entropy_before = _entropy(logits_before)
        loss_before = float(np.mean(net.losses_for_texts(
            texts, tokenizer, batch_size=BATCH_SIZE, max_len=max_len)))
        theta_norm_before = _theta_norm(net)

        # ---- delta generation (frozen P5 mechanism) ----
        delta = net.generate_delta(texts, tokenizer, max_len=max_len)
        delta_norm = float(delta["delta_W_A"].norm()) + \
            float(delta["delta_W_B"].norm())
        d_nan, d_inf = _nan_inf_counts(delta["delta_W_A"], delta["delta_W_B"])
        pooled_h = delta["components"]["pooled_h"]
        hidden_mean = float(pooled_h.mean())
        hidden_std = float(pooled_h.std())

        # ---- parameter transition (frozen P5 method) ----
        before = net.snapshot_parameters()
        net.apply_intrinsic_modification(delta, alpha=alpha)

        # ---- state after the modification (theta_{t+1}) ----
        logits_after = net._logits_for_texts(texts, tokenizer, max_len=max_len)
        entropy_after = _entropy(logits_after)
        loss_after = float(np.mean(net.losses_for_texts(
            texts, tokenizer, batch_size=BATCH_SIZE, max_len=max_len)))
        theta_norm_after = _theta_norm(net)
        applied_change = _applied_change(net, before)
        cumulative_drift = _param_drift(net, theta0)
        t_nan, t_inf = _nan_inf_counts(
            *[p for p in net._current_params_tensors().values()])

        logits_diff = float((logits_after - logits_before).abs().mean())
        pred_change = float((logits_before.argmax(-1) !=
                             logits_after.argmax(-1)).float().mean())

        rec = {
            "round": r + 1,
            "delta_norm": delta_norm,
            "theta_norm": theta_norm_after,
            "relative_delta": delta_norm / max(theta_norm_after, 1e-12),
            "applied_change": applied_change,
            "cumulative_drift": cumulative_drift,
            "logits_diff": logits_diff,
            "pred_change": pred_change,
            "entropy": entropy_after,
            "entropy_before": entropy_before,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss": loss_after,
            "nan_count": d_nan + t_nan,
            "inf_count": d_inf + t_inf,
            "parameter_changed": applied_change > 0.0,
            "hidden_mean": hidden_mean,
            "hidden_std": hidden_std,
            "modulation_strength": delta["modulation_strength"],
        }
        metrics.append(rec)
        if r < 5 or (r + 1) % 25 == 0 or r == NUM_ROUNDS - 1:
            print(f"[p5-long] round {r + 1}: ||d||={delta_norm:.5f} "
                  f"||th||={theta_norm_after:.3f} rel={rec['relative_delta']:.5f} "
                  f"drift={cumulative_drift:.4f} logits={logits_diff:.5f} "
                  f"entropy={entropy_after:.4f} loss={loss_after:.4f} "
                  f"nan={rec['nan_count']} inf={rec['inf_count']}", flush=True)

    # ------------------------------------------------------------------ #
    # summaries
    # ------------------------------------------------------------------ #
    d = np.array([m["delta_norm"] for m in metrics])
    rel = np.array([m["relative_delta"] for m in metrics])
    th = np.array([m["theta_norm"] for m in metrics])
    ld = np.array([m["logits_diff"] for m in metrics])
    ent = np.array([m["entropy"] for m in metrics])
    dr = np.array([m["cumulative_drift"] for m in metrics])
    ls = np.array([m["loss"] for m in metrics])
    nans = sum(m["nan_count"] for m in metrics)
    infs = sum(m["inf_count"] for m in metrics)
    first_nan = next((m["round"] for m in metrics if m["nan_count"] > 0), None)
    first_inf = next((m["round"] for m in metrics if m["inf_count"] > 0), None)

    def window_stats(arr, rounds):
        idx = [m["round"] for m in metrics]
        sel = [arr[i] for i, rr in enumerate(idx) if rounds[0] <= rr <= rounds[1]]
        if not sel:
            return {}
        return {"mean": float(np.mean(sel)), "std": float(np.std(sel)),
                "min": float(np.min(sel)), "max": float(np.max(sel))}

    five_phases = {}
    for a, b in [(1, 30), (31, 60), (61, 90), (91, 120), (121, 150)]:
        five_phases[f"{a}-{b}"] = {
            "delta_norm": window_stats(d, (a, b)),
            "relative_delta": window_stats(rel, (a, b)),
            "theta_norm": window_stats(th, (a, b)),
            "logits_diff": window_stats(ld, (a, b)),
            "entropy": window_stats(ent, (a, b)),
        }

    three_phases = {}
    for name, a, b in [("early", 1, 30), ("middle", 31, 100), ("late", 101, 150)]:
        idx = [i for i, m in enumerate(metrics) if a <= m["round"] <= b]
        three_phases[name] = {
            "mean_delta": float(d[idx].mean()), "std_delta": float(d[idx].std()),
            "mean_relative": float(rel[idx].mean()),
            "drift_start": float(dr[idx[0]]), "drift_end": float(dr[idx[-1]]),
            "mean_logits_diff": float(ld[idx].mean()),
            "mean_entropy": float(ent[idx].mean()),
            "mean_loss": float(ls[idx].mean()),
        }

    def slope(arr):
        x = np.arange(len(arr), dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        return float(np.polyfit(x, arr, 1)[0])

    summary = {
        "git_commit": git_sha(),
        "num_rounds": NUM_ROUNDS,
        "seed": SEED,
        "final_parameter_drift_theta150_minus_theta0": float(dr[-1]),
        "delta_norm": {
            "mean": float(d.mean()), "std": float(d.std()),
            "min": float(d.min()), "max": float(d.max()),
        },
        "relative_delta": {
            "mean": float(rel.mean()), "std": float(rel.std()),
            "min": float(rel.min()), "max": float(rel.max()),
            "final": float(rel[-1]),
        },
        "theta_norm": {"first": float(th[0]), "last": float(th[-1]),
                       "max": float(th.max())},
        "behavioral_drift": {
            "logits_diff_mean": float(ld.mean()),
            "logits_diff_first": float(ld[0]), "logits_diff_last": float(ld[-1]),
            "pred_change_mean": float(np.mean([m["pred_change"] for m in metrics])),
            "entropy_first": float(ent[0]), "entropy_last": float(ent[-1]),
            "loss_first": float(ls[0]), "loss_last": float(ls[-1]),
        },
        "stability": {
            "nan_total": nans, "inf_total": infs,
            "first_nan_round": first_nan, "first_inf_round": first_inf,
            "nan_inf_zero": nans == 0 and infs == 0,
        },
        "persistence": {
            "rounds_with_nonzero_delta": int((d > 1e-12).sum()),
            "min_delta_norm": float(d.min()),
            "delta_trends_to_zero": bool(d[-1] < d[0]),
        },
        "linear_trends_per_round": {
            "delta_norm": slope(d), "theta_norm": slope(th),
            "logits_diff": slope(ld), "entropy": slope(ent),
        },
        "phases_5": five_phases,
        "phases_3": three_phases,
        "wall_seconds": time.time() - t0,
    }

    # ------------------------------------------------------------------ #
    # plots
    # ------------------------------------------------------------------ #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, NUM_ROUNDS + 1)
    plots = {
        "modification_norm_vs_round.png": (x, d, r"$\|\Delta\theta\|_2$",
                                           "Modification magnitude per round"),
        "relative_modification_vs_round.png": (
            x, rel, r"$\|\Delta\theta\|_2 / \|\theta\|_2$",
            "Relative modification per round"),
        "parameter_norm_vs_round.png": (x, th, r"$\|\theta\|_2$",
                                        "Parameter norm per round"),
        "cumulative_parameter_drift.png": (
            x, dr, r"$\|\theta_t - \theta_0\|_2$",
            "Cumulative parameter drift from round 0"),
        "behavioral_drift_vs_round.png": (x, ld, "mean |logits diff|",
                                          "Behavioral drift per round"),
        "entropy_vs_round.png": (x, ent, "output entropy",
                                 "Output entropy per round"),
    }
    for fname, (xx, yy, ylab, title) in plots.items():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xx, yy, lw=0.9)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylab)
        ax.set_title(f"P5 150-round long-horizon — {title}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=140)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(freeze_config(config, base), f, indent=2, default=str)

    print("=== P5 150-round long-horizon summary ===")
    print(f"final drift ||theta150 - theta0|| = {dr[-1]:.5f}")
    print(f"delta_norm: mean={d.mean():.5f} std={d.std():.5f} "
          f"min={d.min():.5f} max={d.max():.5f}")
    print(f"relative: mean={rel.mean():.5f} final={rel[-1]:.5f}")
    print(f"theta_norm: {th[0]:.4f} -> {th[-1]:.4f} (max {th.max():.4f})")
    print(f"NaN={nans} Inf={infs} | first NaN round={first_nan} "
          f"first Inf round={first_inf}")
    print(f"logits_diff mean={ld.mean():.6f} | entropy {ent[0]:.4f}->{ent[-1]:.4f}")
    print(f"phases: " + "; ".join(
        f"{k}: d={v['mean_delta']:.5f}+-{v['std_delta']:.5f} "
        f"rel={v['mean_relative']:.5f} drift {v['drift_start']:.4f}->"
        f"{v['drift_end']:.4f}" for k, v in three_phases.items()))
    print(f"trends/round: delta={slope(d):.3e} theta={slope(th):.3e} "
          f"logits={slope(ld):.3e} entropy={slope(ent):.3e}")
    print("saved to", OUT_DIR)


def freeze_config(config, base) -> dict:
    import torch

    return {
        "git_commit": git_sha(),
        "experiment": "P5 150-round long-horizon intrinsic self-modification",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": SEED,
        "num_rounds": NUM_ROUNDS,
        "batch_size_per_round": BATCH_SIZE,
        "model": {
            "name": config.model_name,
            "hidden_dim": base.hidden_dim,
            "lora_r": config.lora_r, "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        },
        "plasticity": {
            "alpha": config.plasticity_alpha,
            "rank": config.plasticity_rank,
            "meta_dim": config.meta_dim,
            "modulation_strength_init": config.modulation_strength_init,
            "use_hidden": config.use_hidden,
            "use_param_stats": config.use_param_stats,
            "use_meta": config.use_meta,
        },
        "data": {
            "domains": ["general", "math", "logic", "code", "science"],
            "sampling": "with replacement, 20-round phase pattern repeated",
            "max_len": config.max_len,
        },
        "loop": {
            "network": "N1 (single theta-chain)",
            "apply_every_round": True,
            "validation_rollback": False,
            "note": "stress test of the intrinsic mechanism; "
                    "controller-side validation is intentionally disabled",
        },
    }


if __name__ == "__main__":
    main()
