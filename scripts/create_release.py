"""Create GitHub Release for v0.6.0."""
import json, os, sys, urllib.request, urllib.error

REPO = "Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System"
TAG = "v0.6.0"

BODY = """## Major Milestone

DSCNS v0.6.0 achieves **Strong Pass** (RFR_Full < RFR_NoMemory) where v0.5.3 failed. The outcome-directed reward mechanism is the critical difference.

## What Changed

- **5 new modules**: policy_trace, outcome_policy_learning, modification_guard, checkpoint_manager, relay_manager
- **Outcome-directed reward**: R = w_p*dPerf + w_e*dErr + w_s*Drift + w_c*Consistency
- **Safety envelope**: Risk-based magnitude scaling (never sets m=0)
- **Adaptive exploration**: epsilon varies with policy uncertainty
- **Best/Final/Relay checkpoints**: Three different model states per experiment
- **11 experimental conditions**: FullPolicy, NoMemory, FrozenPolicy, RandomMemory, ZeroMemory, NoCredit, NoAlternatives, NoExploration, NoOutcomeReward, Oracle, Random
- **24,750 total rounds**: 11 conditions x 5 seeds x 450 rounds

## Acceptance Criteria

| Criterion | v0.5.3 | v0.6.0 |
|---|---|---|
| Minimum Pass (D_policy > 0) | PASS | **PASS** |
| Mechanism Pass (target_acc > chance) | N/A | **PASS** (0.439 > 0.333) |
| Strong Pass (RFR_Full < RFR_NoMem) | FAIL | **PASS** (0.410 < 0.445) |
| Full Pass (+ EAR > 0) | FAIL | FAIL (EAR=0) |

## Key Results

| Condition | RFR | Target_Acc |
|---|---|---|
| FullPolicy | 0.410 | 0.439 |
| NoMemory | 0.445 | 0.506 |
| RandomMemory | 0.261 | 0.454 |
| FrozenPolicy | 0.472 | 0.843 |
| Random | 0.455 | 0.230 |

## Reproduction

```bash
python scripts/run_phase6.py --smoke
python scripts/demo_inference.py
python scripts/run_phase6.py --rounds 450 --seeds 5
```

## Historical Tags Preserved

v0.5.1, v0.5.2, v0.5.3 all preserved. No history rewritten.
"""

def create_release():
    data = json.dumps({
        "tag_name": TAG,
        "name": "v0.6.0 - Phase 6: Self-Modification Policy Causality & Long-Horizon Relay Learning",
        "body": BODY,
        "draft": False,
        "prerelease": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases",
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"Release created: {result['html_url']}")
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}")
        return None

if __name__ == "__main__":
    create_release()
