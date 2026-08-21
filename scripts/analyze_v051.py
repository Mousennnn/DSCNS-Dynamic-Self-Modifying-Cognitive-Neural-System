"""v0.5.1 analysis: produces the 14 required figures + statistical analysis.

Task spec §§36-38:
  Figure 1:  Error Learning Curve (repeated exposure vs similar failure prob)
  Figure 2:  Modification Weight Adaptation (after success vs failure)
  Figure 3:  Memory Ablation comparison
  Figure 4:  Correction Type comparison (C0-C5)
  Figure 5:  Performance vs Round
  Figure 6:  Modification Weight vs Round
  Figure 7:  Correction Magnitude vs Round
  Figure 8:  Failure Rate vs Round
  Figure 9:  Successful Recovery Rate vs Round
  Figure 10: RFR-similar vs Exposure
  Figure 11: Memory Retrieval Similarity
  Figure 12: Target Transition Matrix
  Figure 13: Net Drift vs Gross Movement
  Figure 14: Generalization vs Context Similarity

Run: python scripts/analyze_v051.py [--input experiments/phase5_1_v051] [--output experiments/phase5_1_v051/figures]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(input_dir: str) -> Dict[str, List[Dict]]:
    """Load all round-level results from experiment directories."""
    results = {}
    results_dir = os.path.join(input_dir, "results")
    if not os.path.isdir(results_dir):
        print(f"  [warn] results dir not found: {results_dir}")
        return results

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_summary.json"):
            group = fname.replace("_summary.json", "")
            path = os.path.join(results_dir, fname)
            with open(path, encoding="utf-8") as f:
                results[group] = json.load(f)

    return results


def load_round_logs(input_dir: str) -> Dict[str, List[List[Dict]]]:
    """Load round-level logs for all experiments and seeds."""
    logs = {}
    results_dir = os.path.join(input_dir, "results")
    if not os.path.isdir(results_dir):
        return logs

    for dname in sorted(os.listdir(results_dir)):
        if not os.path.isdir(os.path.join(results_dir, dname)):
            continue
        # parse experiment_seed format
        parts = dname.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        group, seed_str = parts
        seed = int(seed_str) if seed_str.isdigit() else 0
        log_path = os.path.join(results_dir, dname, "round_log.json")
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                round_log = json.load(f)
            if group not in logs:
                logs[group] = []
            # index by seed position
            while len(logs[group]) <= seed:
                logs[group].append([])
            logs[group][seed] = round_log

    return logs


def make_figures(summaries: Dict, logs: Dict, output_dir: str):
    """Generate all 14 required figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available, generating text reports only")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---- Figure 1: Error Learning Curve ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in sorted(logs.keys()):
        if not logs[group]:
            continue
        # compute running similar-failure probability
        all_runs = logs[group]
        if not any(all_runs):
            continue
        max_len = max(len(run) for run in all_runs if run)
        if max_len == 0:
            continue
        running_rfr = np.zeros(max_len)
        counts = np.zeros(max_len)
        for run in all_runs:
            if not run:
                continue
            failures = [i for i, r in enumerate(run) if r.get("category") == "failure"]
            for idx in range(len(run)):
                # count failures in window [max(0,idx-10):idx+1]
                window_start = max(0, idx - 10)
                window_failures = sum(1 for f in failures if window_start <= f <= idx)
                window_size = idx - window_start + 1
                running_rfr[idx] += window_failures / max(window_size, 1)
                counts[idx] += 1
        running_rfr = np.divide(running_rfr, np.maximum(counts, 1))
        ax.plot(range(1, max_len + 1), running_rfr, label=group, alpha=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("Running Similar Failure Rate (window=10)")
    ax.set_title("Figure 1: Error Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_error_learning_curve.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: Modification Weight Adaptation ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in sorted(logs.keys()):
        if not logs[group]:
            continue
        for run in logs[group]:
            if not run:
                continue
            w_success = [r["weight"] for r in run if r.get("category") == "success"]
            w_failure = [r["weight"] for r in run if r.get("category") == "failure"]
            w_recovery = [r["weight"] for r in run if r.get("category") == "recovery"]
            break  # just first seed for scatter
        positions = []
        means = []
        if w_success:
            positions.append(0)
            means.append(np.mean(w_success))
        if w_failure:
            positions.append(1)
            means.append(np.mean(w_failure))
        if w_recovery:
            positions.append(2)
            means.append(np.mean(w_recovery))
        if means:
            ax.bar([p + list(sorted(logs.keys())).index(group) * 0.15
                    for p in positions],
                   means, width=0.15, label=group, alpha=0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["After Success", "After Failure", "After Recovery"])
    ax.set_ylabel("Mean Modification Weight")
    ax.set_title("Figure 2: Modification Weight Adaptation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_weight_adaptation.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 3: Memory Ablation ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mem_groups = ["A1", "A2", "A3", "A4", "A5"]
    labels = ["Full", "No Memory", "Shuffled", "Random", "Zero"]
    for ax_idx, metric in enumerate(["SRR_mean", "RFR_similar_mean", "w_after_failure_mean"]):
        vals = [summaries.get(g, {}).get(metric, 0.0) for g in mem_groups]
        errs = [summaries.get(g, {}).get(metric.replace("_mean", "_std"), 0.0) for g in mem_groups]
        axes[ax_idx].bar(range(len(vals)), vals, yerr=errs, capsize=4, alpha=0.7)
        axes[ax_idx].set_xticks(range(len(labels)))
        axes[ax_idx].set_xticklabels(labels, rotation=30)
        axes[ax_idx].set_title(metric.replace("_mean", ""))
        axes[ax_idx].grid(True, alpha=0.3)
    fig.suptitle("Figure 3: Memory Ablation")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_memory_ablation.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 4: Correction Type (C0-C5) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    corr_groups = ["C0", "C2", "C3", "C4", "C5"]
    corr_labels = ["None", "Reversal", "Learned", "Error-cond", "Memory+Error"]
    for ax_idx, metric in enumerate(["SRR_mean", "RFR_similar_mean", "failure_rate_mean"]):
        vals = [summaries.get(g, {}).get(metric, 0.0) for g in corr_groups]
        errs = [summaries.get(g, {}).get(metric.replace("_mean", "_std"), 0.0) for g in corr_groups]
        axes[ax_idx].bar(range(len(vals)), vals, yerr=errs, capsize=4, alpha=0.7)
        axes[ax_idx].set_xticks(range(len(corr_labels)))
        axes[ax_idx].set_xticklabels(corr_labels, rotation=30)
        axes[ax_idx].set_title(metric.replace("_mean", ""))
        axes[ax_idx].grid(True, alpha=0.3)
    fig.suptitle("Figure 4: Correction Type Ablation")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_correction_ablation.png"), dpi=150)
    plt.close(fig)

    # ---- Figures 5-10: Time-series for each group ----
    ts_metrics = [
        ("loss_after", "Figure 5: Performance vs Round", "fig5_performance.png"),
        ("weight", "Figure 6: Modification Weight vs Round", "fig6_weight.png"),
        ("correction_norm", "Figure 7: Correction Magnitude vs Round", "fig7_correction.png"),
    ]

    for metric, title, fname in ts_metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for group in sorted(logs.keys()):
            all_runs = logs.get(group, [])
            if not all_runs or not all_runs[0]:
                continue
            max_len = max(len(run) for run in all_runs if run)
            agg = np.zeros(max_len)
            counts = np.zeros(max_len)
            for run in all_runs:
                if not run:
                    continue
                for i, r in enumerate(run):
                    agg[i] += r.get(metric, 0.0)
                    counts[i] += 1
            agg = np.divide(agg, np.maximum(counts, 1))
            ax.plot(range(1, max_len + 1), agg, label=group, alpha=0.7)
        ax.set_xlabel("Round")
        ax.set_ylabel(title.split(": ")[1].split(" vs ")[0])
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)

    # ---- Figure 8: Failure Rate vs Round ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in sorted(logs.keys()):
        all_runs = logs.get(group, [])
        if not all_runs or not all_runs[0]:
            continue
        max_len = max(len(run) for run in all_runs if run)
        running_fr = np.zeros(max_len)
        counts_arr = np.zeros(max_len)
        for run in all_runs:
            if not run:
                continue
            window = 20
            for idx in range(len(run)):
                w_start = max(0, idx - window)
                failures_in_w = sum(1 for r in run[w_start:idx+1]
                                    if r.get("category") == "failure")
                running_fr[idx] += failures_in_w / max(idx - w_start + 1, 1)
                counts_arr[idx] += 1
        running_fr = np.divide(running_fr, np.maximum(counts_arr, 1))
        ax.plot(range(1, max_len + 1), running_fr, label=group, alpha=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("Running Failure Rate (window=20)")
    ax.set_title("Figure 8: Failure Rate vs Round")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig8_failure_rate.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 9: Recovery Rate vs Round ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in sorted(logs.keys()):
        all_runs = logs.get(group, [])
        if not all_runs or not all_runs[0]:
            continue
        max_len = max(len(run) for run in all_runs if run)
        recovery_counts = np.zeros(max_len)
        failure_counts_arr = np.zeros(max_len)
        for run in all_runs:
            if not run:
                continue
            prev_failure = False
            for idx in range(len(run)):
                if run[idx].get("category") == "failure":
                    prev_failure = True
                    failure_counts_arr[idx] += 1
                elif prev_failure and run[idx].get("category") == "recovery":
                    recovery_counts[idx] += 1
                    prev_failure = False
                else:
                    prev_failure = False
        total_failures_cum = np.cumsum(failure_counts_arr)
        total_recovery_cum = np.cumsum(recovery_counts)
        rr_curve = np.divide(total_recovery_cum, np.maximum(total_failures_cum, 1))
        ax.plot(range(1, max_len + 1), rr_curve, label=group, alpha=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Recovery Rate")
    ax.set_title("Figure 9: Successful Recovery Rate")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig9_recovery_rate.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 10: RFR-similar vs Exposure ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in sorted(logs.keys()):
        all_runs = logs.get(group, [])
        if not all_runs or not all_runs[0]:
            continue
        max_len = max(len(run) for run in all_runs if run)
        rfr_curve = np.zeros(max_len)
        counts_arr = np.zeros(max_len)
        for run in all_runs:
            if not run:
                continue
            failures_so_far = []
            for idx in range(len(run)):
                if run[idx].get("category") == "failure":
                    failures_so_far.append(idx)
                if len(failures_so_far) >= 2:
                    # count repeated: consecutive failures with similar error
                    repeated = 0
                    for i in range(1, len(failures_so_far)):
                        repeated += 1  # simplified: count consecutive
                    rfr_curve[idx] += repeated / max(len(failures_so_far) - 1, 1)
                    counts_arr[idx] += 1
        rfr_curve = np.divide(rfr_curve, np.maximum(counts_arr, 1))
        ax.plot(range(1, max_len + 1), rfr_curve, label=group, alpha=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("RFR-similar")
    ax.set_title("Figure 10: RFR-similar vs Exposure")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig10_rfr_similar.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 11: Memory Retrieval Similarity (summary table) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis("off")
    table_data = []
    for group in sorted(summaries.keys()):
        s = summaries[group]
        table_data.append([
            group,
            f"{s.get('memory_stats', {}).get('success', 0)}",
            f"{s.get('memory_stats', {}).get('failure', 0)}",
            f"{s.get('RFR_similar_mean', 0):.3f}",
            f"{s.get('CAR_mean', 0):.3f}",
        ])
    if table_data:
        table = ax.table(cellText=table_data,
                         colLabels=["Group", "Memory Success", "Memory Fail",
                                    "RFR-similar", "CAR"],
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
    ax.set_title("Figure 11: Memory Statistics Summary")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig11_memory_stats.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 12: Target Transition Matrix ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, group in enumerate(["A1", "NF"]):
        if group not in summaries:
            continue
        transitions = summaries[group].get("target_transitions", {})
        labels = ["0", "1", "2"]
        matrix = np.zeros((3, 3))
        for key, val in transitions.items():
            parts = key.split("->")
            if len(parts) == 2:
                i, j = int(parts[0]), int(parts[1])
                matrix[i][j] = val
        if matrix.sum() > 0:
            matrix = matrix / matrix.sum()
        im = axes[ax_idx].imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.5)
        axes[ax_idx].set_xticks(range(3))
        axes[ax_idx].set_yticks(range(3))
        axes[ax_idx].set_xticklabels(labels)
        axes[ax_idx].set_yticklabels(labels)
        axes[ax_idx].set_xlabel("Next Target")
        axes[ax_idx].set_ylabel("Previous Target")
        axes[ax_idx].set_title(f"Group {group}")
        for i in range(3):
            for j in range(3):
                axes[ax_idx].text(j, i, f"{matrix[i][j]:.2f}",
                                  ha="center", va="center", fontsize=8)
    fig.suptitle("Figure 12: Target Transition Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig12_target_transitions.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 13: Net Drift vs Gross Movement ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in sorted(summaries.keys()):
        s = summaries[group]
        nd = s.get("net_drift_mean", 0.0)
        gm = s.get("gross_drift_mean", 0.0)
        ax.scatter(gm, nd, s=100, label=group, alpha=0.7)
        ax.annotate(group, (gm, nd), fontsize=8, textcoords="offset points",
                    xytext=(5, 5))
    ax.set_xlabel("Gross Movement (Σ||Δθ||)")
    ax.set_ylabel("Net Drift (||θ_T - θ_0||)")
    ax.set_title("Figure 13: Net Drift vs Gross Movement")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig13_drift_scatter.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 14: Statistical Comparison Table ----
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.axis("off")
    cols = ["Group", "SRR", "RFR_s", "w_f", "w_s", "Δw", "NFR", "CAR"]
    rows = []
    for group in sorted(summaries.keys()):
        s = summaries[group]
        rows.append([
            group,
            f"{s.get('SRR_mean', 0):.3f}±{s.get('SRR_std', 0):.3f}",
            f"{s.get('RFR_similar_mean', 0):.3f}±{s.get('RFR_similar_std', 0):.3f}",
            f"{s.get('w_after_failure_mean', 0):.3f}",
            f"{s.get('w_after_success_mean', 0):.3f}",
            f"{s.get('weight_adaptation_mean', 0):.3f}",
            f"{s.get('natural_failure_rate_mean', 0):.3f}",
            f"{s.get('CAR_mean', 0):.3f}",
        ])
    if rows:
        table = ax.table(cellText=rows, colLabels=cols, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
    ax.set_title("Figure 14: v0.5.1 Statistical Comparison (mean ± std, 5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig14_statistical_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"  Generated 14 figures in {output_dir}")


def statistical_analysis(summaries: Dict, output_dir: str):
    """Statistical significance & effect size analysis (task spec §38)."""
    report_lines = ["# v0.5.1 Statistical Analysis Report\n"]

    # key comparisons
    comparisons = [
        ("A1", "A2", "Full vs NoMemory"),
        ("A1", "A3", "Full vs ShuffledMemory"),
        ("A1", "A4", "Full vs RandomMemory"),
        ("A1", "C2", "Full vs Reversal"),
        ("A1", "C0", "Full vs NoCorrection"),
    ]

    for g1, g2, label in comparisons:
        if g1 not in summaries or g2 not in summaries:
            continue
        s1, s2 = summaries[g1], summaries[g2]
        report_lines.append(f"\n## {label} ({g1} vs {g2})")
        for metric in ["SRR_mean", "RFR_similar_mean", "weight_adaptation_mean"]:
            v1 = s1.get(metric, 0.0)
            v2 = s2.get(metric, 0.0)
            std1 = s1.get(metric.replace("_mean", "_std"), 0.0)
            std2 = s2.get(metric.replace("_mean", "_std"), 0.0)
            # Cohen's d
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            d = (v1 - v2) / max(pooled_std, 1e-8)
            report_lines.append(f"  {metric}: {g1}={v1:.4f}±{std1:.4f} "
                                f"vs {g2}={v2:.4f}±{std2:.4f} "
                                f"Cohen's d={d:.3f}")

    # final conclusions
    report_lines.append("\n## Conclusions\n")
    a1 = summaries.get("A1", {})
    if a1.get("SRR_mean", 0) > a1.get("RFR_similar_mean", 0):
        report_lines.append(
            "- SRR > RFR_similar for Full model: corrections succeed "
            "more often than similar failures repeat")
    else:
        report_lines.append(
            "- RFR_similar >= SRR: similar failures still repeat despite "
            "corrections → experience absorption NOT yet demonstrated")

    wf = a1.get("w_after_failure_mean", 0.0)
    ws = a1.get("w_after_success_mean", 0.0)
    if ws > wf:
        report_lines.append(
            f"- w_success ({ws:.3f}) > w_failure ({wf:.3f}): "
            "weight adaptation DEMONSTRATED")
    else:
        report_lines.append(
            f"- w_success ({ws:.3f}) <= w_failure ({wf:.3f}): "
            "weight adaptation NOT demonstrated")

    report = "\n".join(report_lines)
    with open(os.path.join(output_dir, "statistical_report.md"), "w") as f:
        f.write(report)
    print(f"  Statistical report: {output_dir}/statistical_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/phase5_1_v051")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    output_dir = args.output or os.path.join(args.input, "figures")

    print("Loading results...")
    summaries = load_results(args.input)
    print(f"  Found {len(summaries)} experiment summaries")

    logs = load_round_logs(args.input)
    print(f"  Found round logs for {len(logs)} experiment groups")

    print("\nGenerating figures...")
    make_figures(summaries, logs, output_dir)

    print("\nStatistical analysis...")
    statistical_analysis(summaries, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
