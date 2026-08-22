"""Phase 6 plotting script — generates figures from experiment data.

Usage:
    python scripts/plotting/plot_phase6.py --input experiments/phase6 --output experiments/phase6/figures
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict, List
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, skipping plots")


def load_all_results(base_dir):
    raw_dir = os.path.join(base_dir, "raw")
    results = {}
    if not os.path.isdir(raw_dir):
        return results
    for seed_dir in sorted(os.listdir(raw_dir)):
        seed_path = os.path.join(raw_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue
        for fname in os.listdir(seed_path):
            if fname.endswith("_result.json"):
                cond = fname.replace("_result.json", "")
                with open(os.path.join(seed_path, fname)) as f:
                    data = json.load(f)
                results.setdefault(cond, []).append(data)
    return results


def load_round_logs(base_dir, max_seeds=5):
    raw_dir = os.path.join(base_dir, "raw")
    logs = {}
    if not os.path.isdir(raw_dir):
        return logs
    for seed_dir in sorted(os.listdir(raw_dir))[:max_seeds]:
        seed_path = os.path.join(raw_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue
        for fname in os.listdir(seed_path):
            if fname.endswith("_round_log.json"):
                cond = fname.replace("_round_log.json", "")
                with open(os.path.join(seed_path, fname)) as f:
                    data = json.load(f)
                logs.setdefault(cond, []).append(data)
    return logs


def plot_training_curve(logs, output_dir):
    """Plot loss trajectory across rounds for each condition."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))
    for (cond, seed_logs), color in zip(sorted(logs.items()), colors):
        if not seed_logs:
            continue
        # average across seeds
        max_round = max(len(sl) for sl in seed_logs)
        all_losses = []
        for sl in seed_logs:
            losses = [r.get("loss_after", 0) for r in sl]
            if len(losses) < max_round:
                losses.extend([losses[-1]] * (max_round - len(losses)))
            all_losses.append(losses)
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        rounds = np.arange(1, len(mean_loss) + 1)
        ax.plot(rounds, mean_loss, label=cond, color=color, linewidth=1.5)
        ax.fill_between(rounds, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.15, color=color)
    ax.set_xlabel("Round")
    ax.set_ylabel("CE Loss")
    ax.set_title("v0.6.0 Training Loss Across Conditions")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v060_p6_training_curve.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "v060_p6_training_curve.svg"))
    plt.close()


def plot_condition_comparison(results, output_dir):
    """Bar chart comparing key metrics across conditions."""
    if not HAS_MPL or not results:
        return
    metrics = ["SRR_mean", "RFR_similar_mean", "EAR_mean",
               "target_accuracy_mean", "magnitude_correlation_mean"]
    conds = sorted(results.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        vals = [results[c].get(metric, 0) for c in conds]
        stds = [results[c].get(f"{metric.replace('_mean', '')}_std", 0)
                for c in conds]
        bars = ax.bar(range(len(conds)), vals, yerr=stds, capsize=3, alpha=0.8)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=7)
        ax.set_title(metric.replace("_mean", ""), fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("v0.6.0 Condition Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v060_p6_condition_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "v060_p6_condition_comparison.svg"))
    plt.close()


def plot_parameter_drift(logs, output_dir):
    """Plot parameter drift over rounds."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))
    for (cond, seed_logs), color in zip(sorted(logs.items()), colors):
        if not seed_logs:
            continue
        max_round = max(len(sl) for sl in seed_logs)
        all_drifts = []
        for sl in seed_logs:
            drifts = [r.get("param_drift", r.get("theta_norm", 0)) for r in sl]
            if len(drifts) < max_round:
                drifts.extend([drifts[-1]] * (max_round - len(drifts)))
            all_drifts.append(drifts)
        mean_d = np.mean(all_drifts, axis=0)
        rounds = np.arange(1, len(mean_d) + 1)
        ax.plot(rounds, mean_d, label=cond, color=color, linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Parameter Drift")
    ax.set_title("v0.6.0 Parameter Drift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v060_p6_parameter_drift.png"), dpi=300)
    plt.close()


def plot_safety_risk(logs, output_dir):
    """Plot safety risk level over rounds."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))
    for (cond, seed_logs), color in zip(sorted(logs.items()), colors):
        if not seed_logs:
            continue
        max_round = max(len(sl) for sl in seed_logs)
        all_risks = []
        for sl in seed_logs:
            risks = [r.get("safety_risk_level", 0) for r in sl]
            if len(risks) < max_round:
                risks.extend([risks[-1]] * (max_round - len(risks)))
            all_risks.append(risks)
        mean_r = np.mean(all_risks, axis=0)
        rounds = np.arange(1, len(mean_r) + 1)
        ax.plot(rounds, mean_r, label=cond, color=color, linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Safety Risk Level")
    ax.set_title("v0.6.0 Safety Risk Over Rounds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v060_p6_safety_risk.png"), dpi=300)
    plt.close()


def plot_exploration_rate(logs, output_dir):
    """Plot exploration rate over rounds."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))
    for (cond, seed_logs), color in zip(sorted(logs.items()), colors):
        if not seed_logs:
            continue
        max_round = max(len(sl) for sl in seed_logs)
        all_eps = []
        for sl in seed_logs:
            eps = [r.get("exploration_eps", 0.15) for r in sl]
            if len(eps) < max_round:
                eps.extend([eps[-1]] * (max_round - len(eps)))
            all_eps.append(eps)
        mean_e = np.mean(all_eps, axis=0)
        rounds = np.arange(1, len(mean_e) + 1)
        ax.plot(rounds, mean_e, label=cond, color=color, linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Exploration Rate (epsilon)")
    ax.set_title("v0.6.0 Adaptive Exploration Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "v060_p6_exploration_rate.png"), dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Phase 6 plotting")
    ap.add_argument("--input", default="experiments/phase6")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not HAS_MPL:
        print("matplotlib not available")
        return

    output_dir = args.output or os.path.join(args.input, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    raw_results = load_all_results(args.input)
    logs = load_round_logs(args.input)

    if not raw_results:
        print("No results found")
        return

    # aggregate per-condition
    agg_keys = ["SRR", "RFR_similar", "EAR", "target_accuracy",
                "magnitude_correlation", "policy_action_mi", "net_drift"]
    results = {}
    for cond, seeds in raw_results.items():
        agg = {}
        for key in agg_keys:
            vals = [s.get(key, 0) for s in seeds if key in s]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        results[cond] = agg

    print(f"Found {len(results)} conditions: {sorted(results.keys())}")

    print("Generating plots...")
    plot_training_curve(logs, output_dir)
    plot_condition_comparison(results, output_dir)
    plot_parameter_drift(logs, output_dir)
    plot_safety_risk(logs, output_dir)
    plot_exploration_rate(logs, output_dir)

    print(f"Figures saved to {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(('.png', '.svg')):
            print(f"  {f}")


if __name__ == "__main__":
    main()
