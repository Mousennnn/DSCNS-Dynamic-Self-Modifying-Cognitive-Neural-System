"""v0.6.0 Release Validation Script.

Checks all required artifacts are present and consistent.
Must PASS before creating GitHub Release.
"""
import json, os, sys

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  [WARN] {name} {detail}")

def main():
    global PASS, FAIL, WARN
    print("=" * 60)
    print("DSCNS v0.6.0 Release Validation")
    print("=" * 60)

    # === A. Version Metadata ===
    print("\n--- A. Version Metadata ---")
    check("CHANGELOG.md exists", os.path.exists("CHANGELOG.md"))
    check("docs/PHASE6.md exists", os.path.exists("docs/PHASE6.md"))
    check("RELEASE_MANIFEST_v0.6.0.json exists", os.path.exists("RELEASE_MANIFEST_v0.6.0.json"))
    
    if os.path.exists("RELEASE_MANIFEST_v0.6.0.json"):
        with open("RELEASE_MANIFEST_v0.6.0.json") as f:
            rm = json.load(f)
        check("manifest version = v0.6.0", rm.get("version") == "v0.6.0")
        check("manifest has acceptance_criteria", "acceptance_criteria" in rm)
        check("manifest has evidence_matrix", "evidence_matrix" in rm)

    # === B. Documentation ===
    print("\n--- B. Documentation ---")
    check("README.md exists", os.path.exists("README.md"))
    check("docs/PHASE6.md exists", os.path.exists("docs/PHASE6.md"))
    check("docs/PHASE6_EVIDENCE.md exists", os.path.exists("docs/PHASE6_EVIDENCE.md"))
    check("docs/LONG_RANGE_RELAY.md exists", os.path.exists("docs/LONG_RANGE_RELAY.md"))
    check("docs/INFERENCE.md exists", os.path.exists("docs/INFERENCE.md"))
    check("docs/RESEARCH_HISTORY.md exists", os.path.exists("docs/RESEARCH_HISTORY.md"))
    check("docs/NEGATIVE_RESULTS.md exists", os.path.exists("docs/NEGATIVE_RESULTS.md"))
    check("experiments/phase6/README.md exists", os.path.exists("experiments/phase6/README.md"))
    check("experiments/phase6/REPORT_v060.md exists", os.path.exists("experiments/phase6/REPORT_v060.md"))

    # === C. Experiment Data ===
    print("\n--- C. Experiment Data ---")
    conds = ["FullPolicy","NoMemory","FrozenPolicy","RandomMemory","ZeroMemory",
             "NoCredit","NoAlternatives","NoExploration","NoOutcomeReward","Oracle","Random"]
    
    total_results = 0
    for cond in conds:
        for si in range(5):
            if os.path.exists(f"experiments/phase6/raw/seed_{si}/{cond}_result.json"):
                total_results += 1
    check(f"All 55 result files present ({total_results}/55)", total_results == 55,
          f"found {total_results}")

    total_rounds = 0
    for cond in conds:
        for si in range(5):
            p = f"experiments/phase6/raw/seed_{si}/{cond}_round_log.json"
            if os.path.exists(p):
                with open(p) as f:
                    logs = json.load(f)
                total_rounds += len(logs)
    check(f"Sufficient round logs (>= 20000)", total_rounds >= 20000,
          f"found {total_rounds} rounds")

    # === D. Summaries ===
    print("\n--- D. Summaries ---")
    for cond in conds:
        check(f"{cond}_summary.json exists",
              os.path.exists(f"experiments/phase6/summaries/{cond}_summary.json"))
    check("analysis_v060.json exists", os.path.exists("experiments/phase6/summaries/analysis_v060.json"))
    check("policy_divergence.json exists", os.path.exists("experiments/phase6/summaries/policy_divergence.json"))
    check("evidence_matrix.json exists", os.path.exists("experiments/phase6/summaries/evidence_matrix.json"))

    # === E. Figures ===
    print("\n--- E. Figures ---")
    fig_dir = "experiments/phase6/figures"
    check("figures/ directory exists", os.path.isdir(fig_dir))
    if os.path.isdir(fig_dir):
        figs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
        check(f"Has PNG figures ({len(figs)})", len(figs) >= 3, f"found {len(figs)}")
    check("figures/README.md exists", os.path.exists(os.path.join(fig_dir, "README.md")))

    # === F. Checkpoints ===
    print("\n--- F. Checkpoints ---")
    check("MANIFEST_v060.json exists", os.path.exists("experiments/phase6/checkpoints/MANIFEST_v060.json"))
    
    best_count = 0
    final_count = 0
    relay_count = 0
    for cond in conds:
        for si in range(5):
            seed = 42 + si
            if os.path.exists(f"experiments/phase6/checkpoints/{cond}/seed_{seed}/best/metadata.json"):
                best_count += 1
            if os.path.exists(f"experiments/phase6/checkpoints/{cond}/seed_{seed}/final/metadata.json"):
                final_count += 1
            if os.path.exists(f"experiments/phase6/relay/{cond}/seed_{seed}/relay_v0.6.0_stage_450/lineage.json"):
                relay_count += 1
    
    check(f"Best checkpoints ({best_count}/55)", best_count == 55)
    check(f"Final checkpoints ({final_count}/55)", final_count == 55)
    check(f"Relay checkpoints ({relay_count}/55)", relay_count == 55)

    # === G. Scripts ===
    print("\n--- G. Scripts ---")
    check("run_phase6.py exists", os.path.exists("scripts/run_phase6.py"))
    check("analyze_phase6.py exists", os.path.exists("scripts/analyze_phase6.py"))
    check("infer.py exists", os.path.exists("scripts/infer.py"))
    check("demo_inference.py exists", os.path.exists("scripts/demo_inference.py"))
    check("plot_phase6.py exists", os.path.exists("scripts/plotting/plot_phase6.py"))
    check("validate_release.py exists", os.path.exists("scripts/validate_release.py"))

    # === H. Tests ===
    print("\n--- H. Tests ---")
    check("test_v060.py exists", os.path.exists("tests/test_v060.py"))
    check("smoke_v060.py exists", os.path.exists("smoke_v060.py"))

    # === I. Demo ===
    print("\n--- I. Demo ---")
    check("demo/ directory exists", os.path.isdir("demo"))
    check("demo/pipeline.json exists", os.path.exists("demo/pipeline.json"))

    # === J. Historical Integrity ===
    print("\n--- J. Historical Integrity ---")
    check("docs/PHASE4.md exists (historical)", os.path.exists("docs/PHASE4.md"))
    check("docs/PHASE5.md exists (historical)", os.path.exists("docs/PHASE5.md"))
    check("docs/PHASE5_1.md exists (historical)", os.path.exists("docs/PHASE5_1.md"))
    check("docs/PHASE5_2.md exists (historical)", os.path.exists("docs/PHASE5_2.md"))
    check("experiments/phase5_3_v053/ exists (historical)", os.path.isdir("experiments/phase5_3_v053"))

    # === K. Scientific Integrity ===
    print("\n--- K. Scientific Integrity ---")
    report_path = "experiments/phase6/REPORT_v060.md"
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        check("Report mentions Full Pass FAIL", "Full Pass" in report and "FAIL" in report)
        check("Report mentions NOT ESTABLISHED", "NOT ESTABLISHED" in report)
    else:
        check("REPORT exists for integrity check", False)

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"Validation Summary: {PASS} passed, {FAIL} failed, {WARN} warnings")
    print(f"{'='*60}")
    
    if FAIL == 0:
        print("\nRELEASE VALIDATION: PASS")
        print("All required artifacts present. Ready for GitHub Release.")
    else:
        print(f"\nRELEASE VALIDATION: FAIL ({FAIL} issues)")
        print("Fix all FAIL items before creating GitHub Release.")
    
    return FAIL

if __name__ == "__main__":
    sys.exit(main())
