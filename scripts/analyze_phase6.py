"""v0.6.0 / Phase 6 Analysis Script.

Full analysis with:
    - Cross-condition policy divergence
    - Causal chain verification
    - Evidence matrix
    - v0.5.3 vs v0.6.0 comparison
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict, List, Optional
import numpy as np

BASE_DIR = os.path.join("experiments", "phase6")


def load_results(base_dir):
    raw_dir = os.path.join(base_dir, "raw")
    sum_dir = os.path.join(base_dir, "summaries")

    results_by_condition = {}
    if not os.path.isdir(raw_dir):
        return results_by_condition, {}

    # collect per-seed results
    for seed_dir in sorted(os.listdir(raw_dir)):
        seed_path = os.path.join(raw_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue
        for fname in os.listdir(seed_path):
            if fname.endswith("_result.json"):
                cond = fname.replace("_result.json", "")
                with open(os.path.join(seed_path, fname)) as f:
                    result = json.load(f)
                results_by_condition.setdefault(cond, []).append(result)

    # load summaries
    summaries = {}
    if os.path.isdir(sum_dir):
        for fname in os.listdir(sum_dir):
            if fname.endswith("_summary.json"):
                with open(os.path.join(sum_dir, fname)) as f:
                    summaries[fname.replace("_summary.json", "")] = json.load(f)

    return results_by_condition, summaries


def aggregate_results(results_by_condition):
    """Aggregate per-seed results with mean +/- std."""
    aggregates = {}
    numeric_keys = [
        "failure_rate", "SRR", "RFR_target", "RFR_similar",
        "w_after_failure", "w_after_success", "weight_adaptation",
        "EAR", "high_sim_failure_rate", "high_sim_success_rate",
        "lineage_efficacy", "net_drift", "gross_drift",
        "credit_mean", "credit_std",
        "experience_value_mean", "experience_value_std",
        "alt_success_rate", "alt_failure_rate",
        "target_accuracy", "magnitude_correlation", "policy_action_mi",
        "confidence_reward_corr", "best_score",
    ]
    for cond, seeds in results_by_condition.items():
        agg = {}
        for key in numeric_keys:
            vals = [s[key] for s in seeds if key in s]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
                agg[f"{key}_ci95"] = float(1.96 * np.std(vals) / max(len(vals), 1) ** 0.5)
        agg["n_seeds"] = len(seeds)
        agg["rounds"] = seeds[0].get("rounds", 450) if seeds else 0
        aggregates[cond] = agg
    return aggregates


def compute_policy_divergence(policy_A_dist, policy_B_dist):
    if not policy_A_dist or not policy_B_dist:
        return {"kl": 0.0, "js": 0.0, "cosine": 0.0}
    n_targets = 3
    def _mean_dist(dists):
        counts = np.zeros(n_targets)
        for d in dists:
            for t, p in d.items():
                counts[min(int(t), n_targets-1)] += p
        counts /= max(len(dists), 1)
        total = counts.sum()
        if total > 0: counts /= total
        else: counts = np.ones(n_targets) / n_targets
        return counts
    p = _mean_dist(policy_A_dist) + 1e-8
    q = _mean_dist(policy_B_dist) + 1e-8
    p /= p.sum(); q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    norm = np.linalg.norm(p) * np.linalg.norm(q)
    cosine = float(np.dot(p, q) / max(norm, 1e-12))
    return {"kl": kl, "js": js, "cosine": cosine}


def policy_divergence_analysis(results_by_condition):
    """Compute cross-condition policy divergence from policy_target_distribution_mean."""
    policy_dists = {}
    for cond, seeds in results_by_condition.items():
        dists = []
        for s in seeds:
            d = s.get("policy_target_distribution_mean", {})
            if d:
                dists.append({int(k): float(v) for k, v in d.items()})
        if dists:
            policy_dists[cond] = dists

    div_results = {}
    baseline = "FullPolicy"
    if baseline not in policy_dists:
        return div_results

    for cond in policy_dists:
        if cond == baseline:
            continue
        div = compute_policy_divergence(policy_dists[baseline], policy_dists[cond])
        div_results[f"{baseline}_vs_{cond}"] = div
    return div_results


def acceptance_criteria(aggregates, div_results):
    """Evaluate acceptance criteria."""
    full = aggregates.get("FullPolicy", {})
    no_mem = aggregates.get("NoMemory", {})

    kl_nm = div_results.get("FullPolicy_vs_NoMemory", {}).get("kl", 0.0)
    rfr_full = full.get("RFR_similar_mean", 0)
    rfr_nm = no_mem.get("RFR_similar_mean", 0)
    ear = full.get("EAR_mean", 0)
    ta = full.get("target_accuracy_mean", 0)
    mc = full.get("magnitude_correlation_mean", 0)
    mi = full.get("policy_action_mi_mean", 0)

    results = {
        "Minimum Pass": {
            "criterion": "D_policy > 0",
            "result": "PASS" if kl_nm > 0 else "FAIL",
            "value": kl_nm,
        },
        "Mechanism Pass": {
            "criterion": "Target accuracy > chance AND magnitude correlation > 0",
            "result": "PASS" if ta > 1/3 and mc > 0 else "FAIL",
            "target_accuracy": ta,
            "magnitude_correlation": mc,
            "policy_action_mi": mi,
        },
        "Strong Pass": {
            "criterion": "D_policy > 0 AND RFR_Full < RFR_NoMemory",
            "result": "PASS" if kl_nm > 0 and rfr_full < rfr_nm else "FAIL",
            "RFR_Full": rfr_full, "RFR_NoMemory": rfr_nm,
        },
        "Full Pass": {
            "criterion": "Strong Pass + EAR > 0",
            "result": "PASS" if (kl_nm > 0 and rfr_full < rfr_nm and ear > 0) else "FAIL",
            "EAR": ear,
        },
    }
    return results


def evidence_matrix(aggregates, div_results):
    """Generate evidence matrix."""
    full = aggregates.get("FullPolicy", {})
    no_mem = aggregates.get("NoMemory", {})
    frozen = aggregates.get("FrozenPolicy", {})
    random = aggregates.get("Random", {})

    return {
        "Experience -> Policy": {
            "evidence": "D_policy > 0",
            "multi_seed": True,
            "status": "SUPPORTED" if div_results.get("FullPolicy_vs_NoMemory", {}).get("kl", 0) > 0 else "NOT ESTABLISHED",
        },
        "Policy -> Target": {
            "evidence": f"target_accuracy={full.get('target_accuracy_mean', 0):.3f}",
            "multi_seed": True,
            "status": "SUPPORTED" if full.get("target_accuracy_mean", 0) > 1/3 else "PARTIAL",
        },
        "Policy -> Magnitude": {
            "evidence": f"magnitude_corr={full.get('magnitude_correlation_mean', 0):.3f}",
            "multi_seed": True,
            "status": "SUPPORTED" if full.get("magnitude_correlation_mean", 0) > 0.1 else "WEAK",
        },
        "Modification -> Outcome": {
            "evidence": f"RFR_Full={full.get('RFR_similar_mean',0):.3f} vs RFR_NoMem={no_mem.get('RFR_similar_mean',0):.3f}",
            "multi_seed": True,
            "status": "NOT ESTABLISHED" if full.get('RFR_similar_mean', 0) >= no_mem.get('RFR_similar_mean', 0) else "SUPPORTED",
        },
        "Outcome -> Credit": {
            "evidence": f"credit_mean={full.get('credit_mean', 0):.4f}",
            "multi_seed": True,
            "status": "PARTIAL",
        },
        "Credit -> Policy": {
            "evidence": f"EAR={full.get('EAR_mean', 0):.4f}",
            "multi_seed": True,
            "status": "NOT ESTABLISHED" if full.get('EAR_mean', 0) <= 0 else "SUPPORTED",
        },
        "Full Closed Loop": {
            "evidence": "all links combined",
            "multi_seed": True,
            "status": "NOT ESTABLISHED",
        },
    }


def generate_report(aggregates, div_results, criteria, evidence, output_path):
    """Generate markdown report."""
    lines = []
    lines.append("# DSCNS v0.6.0 / Phase 6 Analysis Report\n")
    lines.append("## Acceptance Criteria\n")
    for name, info in criteria.items():
        status = info["result"]
        lines.append(f"- **{name}**: {status} — {info['criterion']}")
    lines.append("")

    lines.append("## Evidence Matrix\n")
    lines.append("| Causal Link | Evidence | Status |")
    lines.append("|---|---|---|")
    for link, info in evidence.items():
        lines.append(f"| {link} | {info['evidence']} | {info['status']} |")
    lines.append("")

    lines.append("## Condition Comparison\n")
    lines.append("| Condition | SRR | RFR_similar | EAR | Target_Acc | Mag_Corr | Policy_MI | Net_Drift |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cond, agg in sorted(aggregates.items()):
        lines.append(
            f"| {cond} "
            f"| {agg.get('SRR_mean', 0):.3f} "
            f"| {agg.get('RFR_similar_mean', 0):.3f} "
            f"| {agg.get('EAR_mean', 0):.3f} "
            f"| {agg.get('target_accuracy_mean', 0):.3f} "
            f"| {agg.get('magnitude_correlation_mean', 0):.3f} "
            f"| {agg.get('policy_action_mi_mean', 0):.4f} "
            f"| {agg.get('net_drift_mean', 0):.1f} |")
    lines.append("")

    lines.append("## Policy Divergence (FullPolicy vs each condition)\n")
    lines.append("| Pair | KL | JS | Cosine |")
    lines.append("|---|---|---|---|")
    for pair, div in sorted(div_results.items()):
        lines.append(f"| {pair} | {div['kl']:.4f} | {div['js']:.4f} | {div['cosine']:.4f} |")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description="v0.6.0 / Phase 6 analysis")
    ap.add_argument("--dir", default=BASE_DIR)
    args = ap.parse_args()

    print("Loading results...")
    results, summaries = load_results(args.dir)
    if not results:
        print("No results found. Run experiments first.")
        return

    print("Aggregating...")
    aggregates = aggregate_results(results)

    print("Policy divergence analysis...")
    div_results = policy_divergence_analysis(results)

    print("Acceptance criteria...")
    criteria = acceptance_criteria(aggregates, div_results)

    print("Evidence matrix...")
    evidence = evidence_matrix(aggregates, div_results)

    # save analysis
    analysis = {
        "aggregates": aggregates,
        "policy_divergence": div_results,
        "acceptance_criteria": criteria,
        "evidence_matrix": evidence,
    }
    sum_dir = os.path.join(args.dir, "summaries")
    os.makedirs(sum_dir, exist_ok=True)
    with open(os.path.join(sum_dir, "analysis_v060.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # generate report
    report_path = os.path.join(args.dir, "REPORT_v060.md")
    generate_report(aggregates, div_results, criteria, evidence, report_path)

    # print summary
    print(f"\n{'='*70}")
    print("v0.6.0 / Phase 6 Analysis Summary")
    print(f"{'='*70}")
    for cond, agg in sorted(aggregates.items()):
        print(f"  {cond:18s}: SRR={agg.get('SRR_mean', 0):.3f} "
              f"RFR={agg.get('RFR_similar_mean', 0):.3f} "
              f"EAR={agg.get('EAR_mean', 0):.3f} "
              f"TAcc={agg.get('target_accuracy_mean', 0):.3f}")

    print(f"\nAcceptance Criteria:")
    for name, info in criteria.items():
        print(f"  {name}: {info['result']}")


if __name__ == "__main__":
    main()
