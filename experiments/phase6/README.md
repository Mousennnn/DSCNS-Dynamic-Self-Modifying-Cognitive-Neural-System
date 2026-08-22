# Phase 6 Experiments

## Experiment Overview

DSCNS v0.6.0 Phase 6: **Closed-Loop Self-Modification Investigation**

- **11 experimental conditions** × **5 seeds** × **450 rounds** = **24,750 total rounds**
- Base model: GPT-2 small (124M) + per-network LoRA (r=16, α=32)
- Seeds: 42, 43, 44, 45, 46

## Conditions

| # | Condition | Description | Key Ablation |
|---|---|---|---|
| 1 | **FullPolicy** | Complete v0.6.0 system | — |
| 2 | **NoMemory** | Memory retrieval disabled | Experience → Policy |
| 3 | **FrozenPolicy** | Policy not updated | Policy learning |
| 4 | **RandomMemory** | Memory retrieval randomized | Memory quality |
| 5 | **ZeroMemory** | Memory replaced with zeros | Memory content |
| 6 | **NoCredit** | Temporal credit disabled | Credit assignment |
| 7 | **NoAlternatives** | Single candidate | Candidate diversity |
| 8 | **NoExploration** | ε = 0 (pure exploitation) | Exploration |
| 9 | **NoOutcomeReward** | Outcome-directed reward disabled | Outcome learning |
| 10 | **Oracle** | Oracle policy (upper bound) | Upper bound |
| 11 | **Random** | Random policy (lower bound) | Lower bound |

## Results Summary

| Condition | SRR | RFR | EAR | Target_Acc |
|---|---|---|---|---|
| FullPolicy | 0.000 | 0.410 | 0.000 | 0.439 |
| NoMemory | 0.000 | 0.445 | 0.000 | 0.506 |
| FrozenPolicy | 0.000 | 0.472 | 0.000 | 0.843 |
| Random | 0.000 | 0.455 | 0.000 | 0.230 |

**Strong Pass:** RFR_Full (0.410) < RFR_NoMemory (0.445) ✓

## Directory Structure

```
experiments/phase6/
├── README.md                          # This file
├── REPORT_v060.md                     # Full analysis report
│
├── checkpoints/                       # Best/Final checkpoints
│   ├── MANIFEST_v060.json            # Complete checkpoint inventory
│   ├── {Condition}/
│   │   └── seed_{42-46}/
│   │       ├── best/
│   │       │   ├── model.pt          # Best checkpoint weights
│   │       │   └── metadata.json     # Best score, round, SHA256
│   │       └── final/
│   │           ├── model.pt          # Final checkpoint weights
│   │           └── metadata.json     # Final state metadata
│
├── figures/                           # Generated figures
│   ├── README.md                     # Figure index with descriptions
│   ├── v060_p6_condition_comparison.png
│   ├── v060_p6_training_curve.png
│   ├── v060_p6_parameter_drift.png
│   ├── v060_p6_safety_risk.png
│   └── v060_p6_exploration_rate.png
│
├── raw/                               # Raw experiment data
│   ├── seed_0/                       # Seed 42
│   │   ├── {Condition}_result.json   # Summary metrics
│   │   └── {Condition}_round_log.json # Per-round details
│   ├── seed_1/                       # Seed 43
│   ├── seed_2/                       # Seed 44
│   ├── seed_3/                       # Seed 45
│   └── seed_4/                       # Seed 46
│
├── relay/                             # Relay checkpoints
│   └── {Condition}/
│       └── seed_{42-46}/
│           └── relay_v0.6.0_stage_450/
│               ├── lineage.json      # Cross-version lineage
│               ├── metadata.json     # Relay metadata
│               └── ...              # Model state, config, etc.
│
└── summaries/                         # Aggregated results
    ├── {Condition}_summary.json      # Per-condition aggregate
    ├── analysis_v060.json            # Full analysis output
    ├── evidence_matrix.json          # Causal evidence matrix
    └── policy_divergence.json        # Cross-condition D_policy
```

## Reproduction

```bash
# Run full experiment
python scripts/run_phase6.py --rounds 450 --seeds 5

# Run analysis
python scripts/analyze_phase6.py --dir experiments/phase6

# Generate figures
python scripts/plotting/plot_phase6.py --input experiments/phase6

# Validate release
python scripts/validate_release.py
```

## Report

See [REPORT_v060.md](REPORT_v060.md) for the complete analysis report.
