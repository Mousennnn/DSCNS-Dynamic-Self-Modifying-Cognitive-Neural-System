"""Analyze the P5 long-horizon experiment v2.

Produces the 8 required figures, the summary metrics table, the regime
classification of the 3000-round run, and the Level 0-5 judgment.

Figures (experiments/p5_long_horizon/plots/):
  01_true_net_drift_vs_round.png
  02_gross_movement_vs_round.png
  03_drift_ratio_vs_round.png
  04_parameter_norm_vs_round.png
  05_probe_output_drift_vs_round.png
  06_layer_wise_drift.png
  07_p5_vs_random_control.png
  08_p5_3000_trajectory.png
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BASE = os.path.join("experiments", "p5_long_horizon")
RESULTS = os.path.join(BASE, "results")
PLOTS = os.path.join(BASE, "plots")


def load_metrics(group):
    with open(os.path.join(RESULTS, group, "metrics.json"),
              encoding="utf-8") as f:
        return json.load(f)


def load_group(group):
    with open(os.path.join(RESULTS, group, "group.json"),
              encoding="utf-8") as f:
        return json.load(f)


def arr(rows, key, default=0.0):
    return np.array([r.get(key, default) for r in rows if r.get("round", 0) > 0])


def slope(xs):
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(xs), dtype=float), xs, 1)[0])


def plot(rows_map, out_name, key, ylabel, title, groups=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups = groups or list(rows_map.keys())
    for g in groups:
        if g not in rows_map:
            continue
        xs = arr(rows_map[g], "round")
        ys = arr(rows_map[g], key)
        ax.plot(xs, ys, lw=1.0, label=g)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, out_name), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    groups_150 = ["p5_150", "random_control_150", "no_modification_150"]
    rm = {g: load_metrics(g) for g in groups_150}
    has_3000 = os.path.exists(os.path.join(RESULTS, "p5_3000", "metrics.json"))
    if has_3000:
        rm["p5_3000"] = load_metrics("p5_3000")

    # ---- Figures 1-5 (150-round groups) ----
    plot(rm, "01_true_net_drift_vs_round.png", "net_drift",
         r"$D_{net}(t)=\|\theta_t-\theta_0\|_2$",
         "True net parameter drift per round (150-round groups)")
    plot(rm, "02_gross_movement_vs_round.png", "gross_drift",
         r"$D_{gross}(t)=\sum\|\theta_s-\theta_{s-1}\|_2$",
         "Gross parameter movement per round (150-round groups)")
    plot(rm, "03_drift_ratio_vs_round.png", "drift_ratio",
         r"$R=D_{net}/(D_{gross}+\epsilon)$",
         "Drift ratio per round (150-round groups)")
    plot(rm, "04_parameter_norm_vs_round.png", "theta_norm",
         r"$\|\theta_t\|_2$",
         "Parameter norm per round (150-round groups)")
    plot(rm, "05_probe_output_drift_vs_round.png", "output_drift_vs_0",
         r"$D_{output}(t)=mean|\mathrm{logits}_t-\mathrm{logits}_0|$",
         "Probe output drift vs t=0 (frozen Probe Set)")

    # ---- Figure 6: layer-wise drift (p5_150) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(RESULTS, "p5_150", "layer_drift.json"),
              encoding="utf-8") as f:
        layer_curve = json.load(f)
    rounds = sorted(int(k) for k in layer_curve.keys())
    layer_ids = sorted({int(k) for k in layer_curve[str(rounds[-1])].keys()})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for li in layer_ids:
        dr = np.array([layer_curve[str(r)][str(li)]["drift"] for r in rounds])
        axes[0].plot(rounds, dr, lw=1.0, label=f"layer {li}")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("layer drift")
    axes[0].set_title("Layer-wise net drift (p5_150)")
    axes[0].legend(fontsize=7, ncol=3)
    for li in layer_ids:
        rel = np.array([layer_curve[str(r)][str(li)]["rel"] for r in rounds])
        axes[1].plot(rounds, rel, lw=1.0, label=f"layer {li}")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("relative drift")
    axes[1].set_title("Layer-wise relative drift (p5_150)")
    axes[1].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "06_layer_wise_drift.png"), dpi=140)
    plt.close(fig)

    # ---- Figure 7: P5 vs Random-Control ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for j, (key, ylab) in enumerate([
            ("net_drift", "net drift"), ("gross_drift", "gross movement"),
            ("output_drift_vs_0", "probe output drift")]):
        for g in ["p5_150", "random_control_150"]:
            axes[j].plot(arr(rm[g], "round"), arr(rm[g], key), lw=1.0, label=g)
        axes[j].set_xlabel("Round")
        axes[j].set_ylabel(ylab)
        axes[j].set_title(f"P5 vs Random-Control: {ylab}")
        axes[j].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "07_p5_vs_random_control.png"), dpi=140)
    plt.close(fig)

    # ---- summary metrics table ----
    summary = {}
    for g in groups_150 + (["p5_3000"] if has_3000 else []):
        gr = load_group(g)
        rows = load_metrics(g)
        dd = arr(rows, "delta_norm")
        rl = arr(rows, "drift_ratio")
        od = arr(rows, "output_drift_vs_0")
        summary[g] = {
            "num_rounds": gr["num_rounds"],
            "hash_constant": gr["hash_constant"],
            "param_norm_0": gr["param_norm_0"],
            "param_norm_final": gr["param_norm_final"],
            "gross_parameter_movement": gr["gross_movement"],
            "true_net_parameter_drift": gr["true_net_drift"],
            "drift_ratio": gr["drift_ratio"],
            "max_single_step_drift": gr["max_single_step_drift"],
            "mean_single_step_drift": gr["mean_single_step_drift"],
            "mean_delta_norm": float(dd.mean()) if len(dd) else 0.0,
            "std_delta_norm": float(dd.std()) if len(dd) else 0.0,
            "max_delta_norm": float(dd.max()) if len(dd) else 0.0,
            "min_delta_norm": float(dd.min()) if len(dd) else 0.0,
            "mean_drift_ratio": float(rl.mean()) if len(rl) else 0.0,
            "final_drift_ratio": float(rl[-1]) if len(rl) else 0.0,
            "probe_output_drift_final": gr["probe_output_drift_final"],
            "probe_output_drift_mean": gr["probe_output_drift_mean"],
            "nan_total": sum(r.get("nan_count", 0) for r in rows),
            "inf_total": sum(r.get("inf_count", 0) for r in rows),
            "first_nan_round": gr["first_nan_round"],
            "first_inf_round": gr["first_inf_round"],
        }
    with open(os.path.join(BASE, "analysis_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Figure 8: 3000-round trajectory ----
    if has_3000:
        r3 = load_metrics("p5_3000")
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
        panels = [
            ("delta_norm", "||delta_theta||", 0, 0),
            ("theta_norm", "||theta||", 0, 1),
            ("net_drift", "net drift", 0, 2),
            ("gross_drift", "gross movement", 1, 0),
            ("drift_ratio", "drift ratio", 1, 1),
            ("output_drift_vs_0", "probe output drift", 1, 2),
        ]
        for key, ylab, i, j in panels:
            xs = arr(r3, "round")
            ys = arr(r3, key)
            axes[i][j].plot(xs, ys, lw=0.8)
            axes[i][j].set_xlabel("Round")
            axes[i][j].set_ylabel(ylab)
            axes[i][j].set_title(f"3000-round: {ylab}")
        fig.suptitle("P5-OM-3000 trajectory (extreme long-horizon run)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(PLOTS, "08_p5_3000_trajectory.png"), dpi=140)
        plt.close(fig)

    # ---- regime classification (3000) + linear trends ----
    regimes = {}
    for g in groups_150:
        rows = load_metrics(g)
        dnet = arr(rows, "net_drift")
        regimes[g] = {"net_drift_slope": slope(dnet)}
    if has_3000:
        r3 = load_metrics("p5_3000")
        dnet3 = arr(r3, "net_drift")
        th3 = arr(r3, "theta_norm")
        d3 = arr(r3, "delta_norm")
        od3 = arr(r3, "output_drift_vs_0")
        second_half = dnet3[len(dnet3) // 2:]
        first_half = dnet3[:len(dnet3) // 2]
        # simple regime classification from the data
        growth = float(second_half.mean() - first_half.mean())
        slope_th = slope(th3)
        slope_d = slope(d3)
        slope_od = slope(od3)
        collapsed = float(th3[-1]) < 1e-3 or float(np.max(od3)) < 1e-6
        diverged = float(np.isnan(th3).sum()) > 0 or \
            float(th3.max()) > 1e6 or float(np.isinf(th3).sum()) > 0
        if diverged:
            regime = "D: divergence"
        elif collapsed:
            regime = "E: collapse"
        elif abs(growth) < 0.05 * max(1.0, float(first_half.mean())):
            regime = "A: stable / saturation"
        elif slope_th > 0 and slope_d < 0:
            regime = "B: continuous drift (theta grows, delta decays)"
        else:
            regime = "B: continuous drift"
        # oscillation check: autocorrelation sign flips of delta_norm
        d_centered = d3 - d3.mean()
        if len(d_centered) > 4:
            corr1 = float(np.corrcoef(d_centered[:-1], d_centered[1:])[0, 1])
        else:
            corr1 = float("nan")
        regimes["p5_3000"] = {
            "regime": regime,
            "net_drift_slope": slope(dnet3),
            "theta_norm_slope": slope_th,
            "delta_norm_slope": slope_d,
            "output_drift_slope": slope_od,
            "second_half_minus_first_half_growth": growth,
            "delta_lag1_autocorr": corr1,
            "final_net_drift": float(dnet3[-1]),
            "final_theta_norm": float(th3[-1]),
            "final_output_drift": float(od3[-1]) if len(od3) else 0.0,
            "nan_total": sum(r.get("nan_count", 0) for r in r3),
            "inf_total": sum(r.get("inf_count", 0) for r in r3),
        }
    with open(os.path.join(BASE, "regimes.json"), "w",
              encoding="utf-8") as f:
        json.dump(regimes, f, indent=2, default=str)

    # ---- Level 0-5 judgment (design report section 23) ----
    p5 = summary.get("p5_150", {})
    rc = summary.get("random_control_150", {})
    nm = summary.get("no_modification_150", {})
    levels = {}
    levels["L0_no_parameter_change"] = not p5.get("hash_constant", True) \
        and p5.get("true_net_parameter_drift", 0) > 1e-6
    levels["L1_parameter_modification"] = \
        p5.get("mean_delta_norm", 0) > 1e-6 and \
        p5.get("true_net_parameter_drift", 0) > 1e-6
    levels["L2_net_parameter_drift"] = p5.get("drift_ratio", 0) > 0.5
    levels["L3_behavioral_drift"] = \
        p5.get("probe_output_drift_final", 0) > 1e-4 and \
        nm.get("probe_output_drift_final", 1e9) < 1e-6
    levels["L4_distinct_from_random_control"] = \
        p5.get("probe_output_drift_final", 0) > \
        rc.get("probe_output_drift_final", 0) * 1.5 and \
        p5.get("true_net_parameter_drift", 0) > \
        rc.get("true_net_parameter_drift", 0) * 1.5
    levels["L5_long_term_dynamics"] = has_3000 and \
        regimes.get("p5_3000", {}).get("regime", "").startswith(
            ("A", "B", "C", "F"))
    with open(os.path.join(BASE, "levels.json"), "w",
              encoding="utf-8") as f:
        json.dump(levels, f, indent=2)

    # ---- console summary ----
    print("=== P5 long-horizon v2 analysis ===")
    print(f"{'group':22s} {'net':>8s} {'gross':>8s} {'ratio':>7s} "
          f"{'probe_drift':>11s} {'nan':>4s} {'inf':>4s}")
    for g in groups_150 + (["p5_3000"] if has_3000 else []):
        s2 = summary[g]
        print(f"{g:22s} {s2['true_net_parameter_drift']:8.4f} "
              f"{s2['gross_parameter_movement']:8.4f} "
              f"{s2['drift_ratio']:7.4f} "
              f"{s2['probe_output_drift_final']:11.6f} "
              f"{s2['nan_total']:4d} {s2['inf_total']:4d}")
    print("levels:", json.dumps(levels, indent=1))
    if has_3000:
        print("3000 regime:", json.dumps(regimes["p5_3000"], indent=1))
    print("saved:", PLOTS)


if __name__ == "__main__":
    main()
