# Phase 5.1 v0.5.1 — Memory-Conditioned Outcome Learning & Error-Driven Self-Modification

> **Version: v0.5.1** · **Baseline: v0.5.0** · **Status: Under Validation**

## Overview

v0.5.1 is a refinement and completion of the v0.5.0 correction loop. Where v0.5.0 demonstrated that failures produce corrections applied in the next round, v0.5.1 asks the harder question: **does memory of past failures actually change future modification behavior?**

### Core Research Question

> When the model is required to make non-zero parameter modifications every round, and a modification causes performance degradation, can the model:
> 1. Store the failure as a retrievable memory episode
> 2. Retrieve similar past failures when facing similar situations
> 3. Change its modification weight, target, or direction based on past failures
> 4. Reduce the probability of similar failures repeating

### What v0.5.0 Proved

- ✅ Failure → Correction signal generated
- ✅ Correction applied in next round
- ✅ Correction rate ~98% under injection
- ✅ Recovery rate ~76% under injection

### What v0.5.1 Tests

- ❓ Does memory of past failures change future `w_t` (modification weight)?
- ❓ Does `w_failure < w_success` hold? (weight adaptation)
- ❓ Does RFR_similar decrease with repeated exposure? (error learning curve)
- ❓ Is Full > NoMemory in SRR and RFR? (memory is actually used)
- ❓ Is Full > ShuffledMemory/RandomMemory? (memory content matters, not just having one)

## Architecture Changes

### New Modules

| Module | File | Purpose |
|--------|------|---------|
| `MemoryEncoder` | `dscns/memory_encoder.py` | Episode → z_memory (32-dim representation) |
| `CorrectionPolicy` | `dscns/correction_policy.py` | Error+Memory+Proposal → Correction (C0-C5 modes) |
| `CorrectionPolicyWithMemory` | `dscns/correction_policy.py` | Wrapper: retrieval + encoding + policy |
| `MultiSimilarityRetriever` | `dscns/memory_encoder.py` | Weighted multi-similarity retrieval |
| `MemoryPolicyEncoder` | `dscns/memory_encoder.py` | Retrieved episodes → policy signal |
| `ExperienceReplayBuffer` | `dscns/experience_replay.py` | Offline correction training buffer |
| `V051OutcomeEvaluator` | `dscns/modification_outcome.py` | CAR/SRR/RE separated metrics |
| `NaturalFailureDetector` | `dscns/modification_outcome.py` | Detect model's own failures |

### Modified Modules

| Module | Changes |
|--------|---------|
| `modification_memory.py` | Added multi-similarity retrieval, RFR_similar, RFR_exact, weight stats by outcome, target transitions |
| `config.py` | Added v0.5.1 fields (correction mode, memory encoder, probe sizes, recovery thresholds) |

## Full Closed-Loop Architecture

```
Current State θ_t
        ↓
Internal State h_t
        ↓
Retrieve Similar Episodes (Multi-Similarity)
        ↓
Memory Representation M_t
        ↓
Modification Proposal (P5.1 unchanged)
        ↓
Target Head / Magnitude Head / Delta Head
        ↓
Effective Modification: Δθ_t = w_t × Δθ_t^proposal
        ↓
θ'_t = θ_t + Δθ_t
        ↓
Evaluate (Fixed Probe Set)
        ↓
Outcome Classification
        ↓
┌───────────────┬────────────────┐
│ SUCCESS       │ FAILURE        │
│               │                │
│ reinforce     │ ErrorEncoder   │
│ experience    │       ↓        │
│               │ Memory.store   │
│               │       ↓        │
│               │ Correction     │
└───────┬───────┴───────┬────────┘
        │               │
        └───────┬───────┘
                ↓
           Next Round (t+1)
                ↓
      Memory-conditioned Policy
                ↓
      New Modification Weight/Target/Direction
```

## Experiment Design

### Memory Ablation (§21, 5 conditions)

| Group | Mode | Description |
|-------|------|-------------|
| A1 | Full (C5+Memory) | Full model with memory retrieval |
| A2 | No Memory (C4) | Error-conditioned only, memory zeroed |
| A3 | Shuffled Memory | Memory from random episodes (wrong context) |
| A4 | Random Memory | Random embeddings as memory |
| A5 | Zero Memory | Zero vectors as memory |

**Key comparison**: A1 vs A2/A3/A4/A5 → proves memory is used and content matters.

### Correction Ablation (§23, 5 conditions)

| Group | Mode | Description |
|-------|------|-------------|
| C0 | No correction | No correction signal |
| C2 | Pure reversal | -Δθ (fixed rollback in direction) |
| C3 | Learned (no memory) | Corrector without memory input |
| C4 | Error-conditioned | Error signal conditions correction |
| C5 | Error + Memory | Full model (C5) |

### Natural Failure Experiment (§19)

| Group | Description |
|-------|-------------|
| NF | No injection, observe natural failures |

### Controlled Failure Experiment (§20)

| Group | Description |
|-------|-------------|
| CF | Weight corruption at injection rounds + full correction |

## Key Metrics

### Separated Recovery Metrics (§16)

| Metric | Definition | Formula |
|--------|-----------|---------|
| CAR | Correction Application Rate | N_corrections / N_failures |
| SRR | Successful Recovery Rate | N_successful_recovery / N_failures |
| RE | Recovery Efficiency | (P_after_corr - P_after_fail) / (P_before - P_after_fail + ε) |

### Repeat Failure Rate Variants (§17)

| Metric | Definition |
|--------|-----------|
| RFR_target | Same target group repeated |
| RFR_similar | Similar context/error/proposal (cosine > threshold) |
| RFR_exact | Exact same conditions repeated |

### Weight Adaptation (§4)

- `w_after_failure`: mean modification weight after failure rounds
- `w_after_success`: mean modification weight after success rounds
- **Goal**: `w_success > w_failure` → weight adapts to outcomes

## Success Criteria

### Must Achieve
1. ✅ Non-zero modification every round
2. ✅ Correction applied in next round after failure
3. ✅ Memory enters correction path (not zero vectors)
4. ✅ CAR/SRR/RE separated in reporting
5. ✅ Similar failure definition upgraded
6. ✅ 5 seeds
7. ✅ Natural failure experiment
8. ✅ Memory ablation
9. ✅ Correction ablation
10. ✅ Original data preserved

### Strong Success (requires evidence)
- `w_success > w_failure` across 5 seeds
- `SRR_A1 > SRR_A2` (Full > No Memory)
- `RFR_similar_A1 < RFR_similar_A2` (Full model reduces similar failures)
- `RFR_similar_A1 < RFR_similar_A3` (memory content matters)

### Honest Conclusion if Negative
If memory doesn't measurably help:
> "Episodic memory currently does not produce measurable error-experience absorption under these experimental conditions. The DSCNS can perform local error correction but has not demonstrated persistent error-based behavioral adaptation."

This is a valid research finding, not a failure.

## Files

```
dscns/
├── memory_encoder.py          # MemoryEncoder, MultiSimilarityRetriever, ModificationEpisode
├── correction_policy.py       # CorrectionPolicy (C0-C5), CorrectionPolicyWithMemory
├── experience_replay.py       # ExperienceReplayBuffer, train_correction_offline
├── modification_memory.py     # Extended with multi-similarity, RFR variants
├── modification_outcome.py    # V051OutcomeEvaluator, RecoveryMetrics, NaturalFailureDetector
├── config.py                  # v0.5.1 fields added
├── intrinsic_plasticity.py    # Unchanged (P5/P5.1 preserved)
├── correction_generator.py    # Unchanged (P5.2 preserved)
├── error_correction.py        # Unchanged (ErrorState, ErrorEncoder preserved)
├── networks.py                # Unchanged (CognitiveNetwork preserved)
scripts/
├── run_v051.py                # Main experiment runner (all ablations)
├── analyze_v051.py            # 14 figures + statistical analysis
config/
├── phase5_1_v051.yaml         # v0.5.1 configuration
experiments/
├── phase5_1_v051/
│   ├── results/               # Per-seed + aggregated results
│   ├── figures/               # 14 required figures
│   ├── configs/               # Frozen experiment configs
│   └── reports/               # Statistical analysis reports
```

## Backward Compatibility

- P5 (`intrinsic_plasticity.py` forward()) → unchanged
- P5.1 (`generate_proposal()`) → unchanged
- P5.2 (`correction_generator.py`) → unchanged, still usable
- All old experiments (`experiments/phase5/`, `phase5_1/`, `phase5_2/`) → untouched
- All old configs → untouched

## Reproducibility

Each experiment saves:
- Config hash + git commit
- Seed
- Parameter hash (SHA-256)
- Round-level metrics (every round)
- Memory state snapshot
- Full round logs with error states
