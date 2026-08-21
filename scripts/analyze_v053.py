"""v0.5.3 Analysis — Policy Divergence + Full Metrics.

Analyzes experiment results focusing on:
  1. Policy Divergence (D_policy) between conditions
  2. Experience → Policy coupling evidence
  3. All standard metrics (SRR, RFR, EAR, etc.)
  4. Temporal credit effectiveness
  5. Experience value trajectory
  6. Causal chain: E → π → Δθ → O
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict, List, Optional
import numpy as np

BASE_DIR = os.path.join("experiments", "phase5_3_v053")


def load_condition_results(out_dir, condition, n_seeds=5):
    """Load per-seed results for a condition."""
    results = []
    for si in range(n_seeds):
        path = os.path.join(out_dir, "raw", f"seed_{si}", f"{condition}_result.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def load_all_summaries(out_dir):
    """Load all summary JSONs."""
    summaries = {}
    summary_dir = os.path.join(out_dir, "summaries")
    if os.path.exists(summary_dir):
        for fn in os.listdir(summary_dir):
            if fn.endswith(".json"):
                with open(os.path.join(summary_dir, fn)) as f:
                    summaries[fn.replace(".json", "")] = json.load(f)
    return summaries


def load_round_logs(out_dir, condition, n_seeds=5):
    """Load per-seed round logs."""
    logs = []
    for si in range(n_seeds):
        path = os.path.join(out_dir, "raw", f"seed_{si}", f"{condition}_round_log.json")
        if os.path.exists(path):
            with open(path) as f:
                logs.append(json.load(f))
    return logs


def aggregate_seeds(results, keys):
    """Aggregate numeric keys across seeds: mean, std, 95% CI."""
    agg = {}
    for key in keys:
        vals = [r.get(key, 0.0) for r in results if key in r]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
            ci95 = 1.96 * std / max(np.sqrt(len(arr)), 1)
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
            agg[f"{key}_ci95"] = ci95
            agg[f"{key}_values"] = vals
    return agg


def compute_round_level_metrics(round_logs_list, metric_name, window=50):
    """Compute rolling metric over rounds across seeds."""
    if not round_logs_list:
        return {}
    # use first seed's round log structure
    max_round = max(r["round"] for log in round_logs_list for r in log)
    windows = list(range(0, max_round + 1, window))
    rolling = {}
    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        vals = []
        for log in round_logs_list:
            recs = [r for r in log if start < r.get("round", 0) <= end]
            if recs:
                vals.append(np.mean([r.get(metric_name, 0.0) for r in recs]))
        if vals:
            rolling[f"R{start}-{end}"] = float(np.mean(vals))
    return rolling


def policy_divergence_analysis(all_summaries, conditions):
    """Compute policy divergence between FullPolicy and each other condition."""
    print("\n" + "=" * 70)
    print("POLICY DIVERGENCE ANALYSIS")
    print("=" * 70)

    div_results = {}
    if "policy_divergence" in all_summaries:
        div_data = all_summaries["policy_divergence"]
        for pair, metrics in div_data.items():
            kl = metrics.get("kl", 0)
            js = metrics.get("js", 0)
            cosine = metrics.get("cosine", 0)
            print(f"\n  {pair}:")
            print(f"    KL divergence:  {kl:.4f}")
            print(f"    JS divergence:  {js:.4f}")
            print(f"    Cosine sim:     {cosine:.4f}")

            # interpretation
            if kl > 0.01:
                print(f"    → D_policy > 0: Memory IS changing policy")
            else:
                print(f"    → D_policy ≈ 0: Memory NOT changing policy")

            div_results[pair] = metrics

    return div_results


def experience_policy_coupling(all_summaries, conditions):
    """Analyze whether experience changes policy."""
    print("\n" + "=" * 70)
    print("EXPERIENCE → POLICY COUPLING")
    print("=" * 70)

    coupling = {}
    for cond in conditions:
        summary = all_summaries.get(cond, {})
        ear = summary.get("EAR_mean", 0)
        credit = summary.get("credit_mean", 0)
        exp_val = summary.get("experience_value_mean", 0)
        alt_success = summary.get("alt_success_rate_mean", 0)
        print(f"\n  {cond}:")
        print(f"    EAR:               {ear:.4f}")
        print(f"    Credit (mean):     {credit:.4f}")
        print(f"    Exp Value (mean):  {exp_val:.4f}")
        print(f"    Alt Success Rate:  {alt_success:.4f}")
        coupling[cond] = {
            "EAR": ear, "credit": credit,
            "exp_value": exp_val, "alt_success": alt_success}

    return coupling


def modification_change_analysis(all_summaries, conditions):
    """Analyze whether policy change leads to modification change."""
    print("\n" + "=" * 70)
    print("MODIFICATION CHANGE ANALYSIS")
    print("=" * 70)

    mod_change = {}
    for cond in conditions:
        summary = all_summaries.get(cond, {})
        srr = summary.get("SRR_mean", 0)
        rfr = summary.get("RFR_similar_mean", 0)
        wf = summary.get("w_after_failure_mean", 0)
        ws = summary.get("w_after_success_mean", 0)
        drift = summary.get("net_drift_mean", 0)
        print(f"\n  {cond}:")
        print(f"    SRR:        {srr:.4f}")
        print(f"    RFR_sim:    {rfr:.4f}")
        print(f"    w_failure:  {wf:.4f}")
        print(f"    w_success:  {ws:.4f}")
        print(f"    Net drift:  {drift:.4f}")
        mod_change[cond] = {
            "SRR": srr, "RFR": rfr,
            "w_failure": wf, "w_success": ws, "drift": drift}

    return mod_change


def outcome_improvement_analysis(all_summaries, conditions):
    """Analyze whether modification change improves outcomes."""
    print("\n" + "=" * 70)
    print("OUTCOME IMPROVEMENT ANALYSIS")
    print("=" * 70)

    outcomes = {}
    for cond in conditions:
        summary = all_summaries.get(cond, {})
        ear = summary.get("EAR_mean", 0)
        rfr = summary.get("RFR_similar_mean", 0)
        failure_rate = summary.get("failure_rate_mean", 0)
        recovery = summary.get("recovery_rate_mean", 0)
        print(f"\n  {cond}:")
        print(f"    EAR:            {ear:.4f}")
        print(f"    RFR_similar:    {rfr:.4f}")
        print(f"    Failure rate:   {failure_rate:.4f}")
        print(f"    Recovery rate:  {recovery:.4f}")
        outcomes[cond] = {
            "EAR": ear, "RFR": rfr,
            "failure_rate": failure_rate, "recovery": recovery}

    return outcomes


def causal_chain_verification(all_summaries, conditions):
    """Verify the 4-level causal chain: E → π → Δθ → O."""
    print("\n" + "=" * 70)
    print("CAUSAL CHAIN VERIFICATION")
    print("  E → π → Δθ → O")
    print("=" * 70)

    chain = {}

    # Link 1: E → π (Experience changes Policy)
    full_summary = all_summaries.get("FullPolicy", {})
    nomem_summary = all_summaries.get("NoMemory", {})

    kl_full_vs_nomem = 0.0
    if "policy_divergence" in all_summaries:
        div = all_summaries["policy_divergence"].get(
            "FullPolicy_vs_NoMemory", {})
        kl_full_vs_nomem = div.get("kl", 0)

    link1 = kl_full_vs_nomem > 0.001
    print(f"\n  Link 1: E → π (Experience changes Policy)")
    print(f"    D_policy(Full vs NoMemory) KL = {kl_full_vs_nomem:.4f}")
    print(f"    {'✓ PASS' if link1 else '✗ FAIL'}")

    # Link 2: π → Δθ (Policy changes Modification)
    full_alt = full_summary.get("alt_success_rate_mean", 0)
    full_credit = full_summary.get("credit_mean", 0)
    link2 = full_credit != 0  # credit is nonzero if policy is influenced
    print(f"\n  Link 2: π → Δθ (Policy changes Modification)")
    print(f"    Credit mean: {full_credit:.4f}")
    print(f"    Alt success rate: {full_alt:.4f}")
    print(f"    {'✓ PASS' if link2 else '✗ FAIL'}")

    # Link 3: Δθ → O (Modification changes Outcome)
    full_rfr = full_summary.get("RFR_similar_mean", 1.0)
    nomem_rfr = nomem_summary.get("RFR_similar_mean", 1.0)
    link3 = full_rfr < nomem_rfr if nomem_rfr > 0 else False
    print(f"\n  Link 3: Δθ → O (Modification changes Outcome)")
    print(f"    RFR_Full: {full_rfr:.4f} vs RFR_NoMemory: {nomem_rfr:.4f}")
    print(f"    {'✓ PASS' if link3 else '✗ FAIL'}")

    # Link 4: Full chain
    link4 = link1 and link2 and link3
    print(f"\n  Full Chain E → π → Δθ → O:")
    print(f"    {'✓ PASS' if link4 else '✗ PARTIAL (see above)'}")

    chain = {
        "link1_E_to_pi": {"kl": kl_full_vs_nomem, "pass": link1},
        "link2_pi_to_delta": {"credit": full_credit, "pass": link2},
        "link3_delta_to_O": {"rfr_full": full_rfr, "rfr_nomem": nomem_rfr, "pass": link3},
        "full_chain": link4,
    }
    return chain


def acceptance_criteria(all_summaries):
    """Check against the formal acceptance criteria."""
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    full = all_summaries.get("FullPolicy", {})
    nomem = all_summaries.get("NoMemory", {})

    # Minimum Pass: D_policy > 0
    kl = 0.0
    if "policy_divergence" in all_summaries:
        div = all_summaries["policy_divergence"].get("FullPolicy_vs_NoMemory", {})
        kl = div.get("kl", 0)
    min_pass = kl > 0.001
    print(f"\n  Minimum Pass: D_policy > 0")
    print(f"    KL = {kl:.4f}")
    print(f"    {'✓ PASS' if min_pass else '✗ FAIL'}")

    # Strong Pass: D_policy > 0 AND RFR_Full < RFR_NoMemory
    rfr_full = full.get("RFR_similar_mean", 1.0)
    rfr_nomem = nomem.get("RFR_similar_mean", 1.0)
    strong_pass = min_pass and (rfr_full < rfr_nomem if rfr_nomem > 0 else False)
    print(f"\n  Strong Pass: D_policy > 0 AND RFR_Full < RFR_NoMemory")
    print(f"    KL = {kl:.4f}, RFR_full = {rfr_full:.4f}, RFR_nomem = {rfr_nomem:.4f}")
    print(f"    {'✓ PASS' if strong_pass else '✗ FAIL'}")

    # Full Pass: all of above + SuccessReuse + EAR > 0
    ear = full.get("EAR_mean", 0)
    full_pass = strong_pass and ear > 0
    print(f"\n  Full Pass: + EAR > 0")
    print(f"    EAR = {ear:.4f}")
    print(f"    {'✓ PASS' if full_pass else '✗ FAIL'}")

    return {
        "minimum_pass": min_pass, "kl": kl,
        "strong_pass": strong_pass, "rfr_full": rfr_full, "rfr_nomem": rfr_nomem,
        "full_pass": full_pass, "EAR": ear,
    }


def main():
    ap = argparse.ArgumentParser(description="v0.5.3 analysis")
    ap.add_argument("--out", default=BASE_DIR)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    conditions = ["FullPolicy", "NoMemory", "FrozenPolicy", "RandomMemory",
                  "ZeroMemory", "NoCredit", "NoAlternatives", "NoExploration"]

    # load data
    print(f"Loading results from {args.out} ...")
    summaries = load_all_summaries(args.out)

    # analysis
    div_results = policy_divergence_analysis(summaries, conditions)
    coupling = experience_policy_coupling(summaries, conditions)
    mod_change = modification_change_analysis(summaries, conditions)
    outcomes = outcome_improvement_analysis(summaries, conditions)
    chain = causal_chain_verification(summaries, conditions)
    acceptance = acceptance_criteria(summaries)

    # save comprehensive analysis
    analysis = {
        "policy_divergence": div_results,
        "experience_policy_coupling": coupling,
        "modification_change": mod_change,
        "outcome_improvement": outcomes,
        "causal_chain": chain,
        "acceptance_criteria": acceptance,
    }

    analysis_path = os.path.join(args.out, "summaries", "analysis_v053.json")
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {analysis_path}")

    # ---- round-level metrics for each condition ----
    print("\n" + "=" * 70)
    print("ROUND-LEVEL METRICS")
    print("=" * 70)
    for cond in conditions:
        logs = load_round_logs(args.out, cond, args.seeds)
        if not logs:
            continue
        credit_rolling = compute_round_level_metrics(logs, "credit", window=50)
        weight_rolling = compute_round_level_metrics(logs, "weight", window=50)
        exp_val_rolling = compute_round_level_metrics(logs, "experience_value", window=50)
        print(f"\n  {cond}:")
        if credit_rolling:
            print(f"    Credit trajectory:  {credit_rolling}")
        if weight_rolling:
            print(f"    Weight trajectory:  {weight_rolling}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
