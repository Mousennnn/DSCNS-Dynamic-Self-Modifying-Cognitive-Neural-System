"""Phase 5 analysis: comparison tables + plots.

Consumes experiments/phase5/{fixed,p5b,p5c,random,constant,shuffled}.json
and validation.json; writes:
  phase5_comparison.json, phase5_loop.png, phase5_controls.png,
  phase5_perf.png, phase5_learning.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _load(out, name):
    path = os.path.join(out, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table(out):
    rows = []
    for tag in ["fixed", "p5b", "p5c", "random", "constant", "shuffled"]:
        res = _load(out, tag)
        if res is None:
            continue
        cl = res.get("closed_loop", {})
        pn = res.get("param_norm_curve", [])
        pm = np.asarray(res["performance_matrix"])
        rows.append({
            "arm": tag,
            "delta_mode": res.get("delta_mode"),
            "final_mean": float(pm[-1].mean()) if len(pm) else 0.0,
            "AF": res.get("AF", 0.0),
            "FWT": res.get("FWT", 0.0),
            "CLS": res.get("CLS", 0.0),
            "triggers": res.get("triggers", 0),
            "acceptance_rate": res.get("acceptance_rate", 0.0),
            "delta_norm_mean": cl.get("delta_norm_mean"),
            "delta_norm_variance": cl.get("delta_norm_variance"),
            "pred_change_mean": cl.get("pred_change_mean"),
            "logits_diff_mean": cl.get("logits_diff_mean"),
            "param_norm_first": pn[0] if pn else None,
            "param_norm_last": pn[-1] if pn else None,
            "wall_seconds": res.get("wall_seconds", 0.0),
        })
    return rows


def plot_perf(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        res = _load(out, row["arm"])
        if res is None:
            continue
        pm = np.asarray(res["performance_matrix"])
        ax.plot(range(1, len(pm) + 1), pm.mean(axis=1), marker="o",
                markersize=3, label=row["arm"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean performance (exp(-loss))")
    ax.set_title("Phase 5: mean performance per round (descriptive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase5_perf.png"), dpi=140)
    plt.close(fig)


def plot_loop(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = _load(out, "p5b")
    if res is None:
        return
    log = res.get("plasticity_log", [])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    if log:
        ax.plot(range(1, len(log) + 1), [m["delta_norm"] for m in log],
                marker="o", markersize=3)
    ax.set_xlabel("Plasticity event #")
    ax.set_ylabel("||delta_theta||")
    ax.set_title("Generated delta norm per event (p5b)")
    ax2 = axes[1]
    pn = res.get("param_norm_curve", [])
    if pn:
        ax2.plot(range(1, len(pn) + 1), pn, marker="s", markersize=3)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("||theta|| (adapter)")
    ax2.set_title("Parameter norm over rounds")
    ax3 = axes[2]
    strengths = [m["modulation_strength"] for m in log] if log else []
    if strengths:
        ax3.plot(range(1, len(strengths) + 1), strengths, marker="^",
                 markersize=3)
    ax3.set_xlabel("Plasticity event #")
    ax3.set_ylabel("modulation strength")
    ax3.set_title("Learnable modulation strength")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase5_loop.png"), dpi=140)
    plt.close(fig)


def plot_controls(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [r["arm"] for r in rows if r["arm"] != "fixed"]
    acc = [r["acceptance_rate"] for r in rows if r["arm"] != "fixed"]
    pred = [r["pred_change_mean"] or 0.0 for r in rows if r["arm"] != "fixed"]
    dvar = [r["delta_norm_variance"] or 0.0 for r in rows if r["arm"] != "fixed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(arms))
    ax = axes[0]
    ax.bar(x, acc, color=["#1f77b4"] + ["#ff7f0e"] * (len(arms) - 1))
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Safety validation pass rate")
    ax2 = axes[1]
    ax2.bar(x, pred, color=["#1f77b4"] + ["#ff7f0e"] * (len(arms) - 1))
    ax2.set_xticks(x); ax2.set_xticklabels(arms)
    ax2.set_ylabel("Prediction change rate")
    ax2.set_title("Behavioral effect of delta")
    ax3 = axes[2]
    ax3.bar(x, dvar, color=["#1f77b4"] + ["#ff7f0e"] * (len(arms) - 1))
    ax3.set_xticks(x); ax3.set_xticklabels(arms)
    ax3.set_yscale("log")
    ax3.set_ylabel("delta_norm variance")
    ax3.set_title("Delta variability across events")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase5_controls.png"), dpi=140)
    plt.close(fig)


def plot_learning(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = _load(out, "p5c")
    if res is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    curves = []
    for net_id, st in res.get("plasticity_trainer", {}).items():
        if st.get("train_loss_curve"):
            curves.append((net_id, st["train_loss_curve"]))
    for net_id, curve in curves:
        ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=3,
                label=net_id)
    ax.set_xlabel("Training call #")
    ax.set_ylabel("Reward-weighted MSE")
    ax.set_title("P5-C plasticity module training loss")
    if curves:
        ax.legend()
    ax2 = axes[1]
    rewards = res.get("rewards", [])
    if rewards:
        ax2.plot(range(1, len(rewards) + 1), rewards, marker="o", markersize=3)
        ax2.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax2.set_xlabel("Accepted modification #")
    ax2.set_ylabel("Reward (perf_after - perf_before)")
    ax2.set_title("P5-C modification rewards")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase5_learning.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = build_table(args.out)
    with open(os.path.join(args.out, "phase5_comparison.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    plot_perf(args.out, rows)
    plot_loop(args.out)
    plot_controls(args.out, rows)
    plot_learning(args.out)

    print("=== Phase 5 comparison table ===")
    header = (f"{'arm':10s} {'final':>7s} {'AF':>7s} {'FWT':>7s} {'CLS':>7s} "
              f"{'trig':>4s} {'acc':>5s} {'d_norm':>9s} {'pred':>6s}")
    print(header)
    for r in rows:
        print(f"{r['arm']:10s} {r['final_mean']:7.4f} {r['AF']:7.4f} "
              f"{r['FWT']:7.4f} {r['CLS']:7.4f} {r['triggers']:4d} "
              f"{r['acceptance_rate']:5.2f} "
              f"{(r['delta_norm_mean'] or 0.0):9.5f} "
              f"{(r['pred_change_mean'] or 0.0):6.4f}")
    print("plots written to", args.out)


if __name__ == "__main__":
    main()
