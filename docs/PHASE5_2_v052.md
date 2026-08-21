# Phase 5.2 v0.5.2 — Persistent Error-Experience Absorption & Self-Modification Learning

> **Version: v0.5.2** · **Baseline: v0.5.1** · **Status: Under Implementation**

## Overview

v0.5.2 advances from Memory-Assisted Correction (v0.5.1) to Persistent Error-Experience Absorption. Where v0.5.1 showed that memory helps recovery (SRR: 0.065 > 0.025), it could not demonstrate that the model changes how it modifies itself based on past failure experience (RFR_similar ≈ 0.886 across all conditions).

### Core Research Question

> Does the system change how it modifies itself because of what it experienced before?

Specifically:
1. Does failure experience reduce the probability of similar future failures?
2. Does success experience increase the probability of similar future successes?
3. Is the weight outcome-conditioned (w_success > w_failure)?

### What v0.5.1 Proved

- ✅ Memory improves SRR (Full: 0.065 vs NoMemory: 0.025)
- ✅ Correction works (C3/C4/C5 beat C0/C2)
- ❌ RFR_similar ≈ 0.886 across all conditions (no experience absorption)
- ❌ w_failure > w_success (opposite of expected)
- ❌ A3 (Shuffled) = A1 (Full) — memory content didn't differentiate

### What v0.5.2 Must Demonstrate

- RFR_similar,Full < RFR_similar,NoMemory
- w_success > w_failure
- P(FutureSimilarFailure) < P(InitialFailure)
- P(Reuse|Success) > P(Reuse|Failure)
- Memory content actually matters (A3 ≠ A1)

## Key Architecture Changes from v0.5.1

### 1. Trained Memory Encoder
v0.5.1's memory encoder was randomly initialized and never trained during inference, producing identical results regardless of memory content. v0.5.2 trains the encoder online via ranking loss.

### 2. Modification Direction Encoding (§6)
New `DirectionEncoder` encodes actual Δθ direction (not just norm), enabling:
- S_direction = cos(z_direction_current, z_direction_memory)
- Distinguishing +ΔW from -ΔW even when ||ΔW|| is the same

### 3. Outcome-Aware Attention (§7-8)
Replaces mean pooling with learned attention:
- q = current context; k_i = episode (context + error + direction + outcome)
- a_i = softmax(q^T k_i / √d); M = Σ a_i v_i
- SUCCESS and FAILURE episodes are distinguished

### 4. Weight Ranking Loss (§9-10)
Online training enforces: w_success > w_failure + margin
- L_ranking = max(0, margin - w_success + w_failure)
- Prevents the v0.5.1 reversal (w_failure > w_success)

### 5. Online Policy Training
Correction policy + memory encoder receive gradient updates during the experiment via:
- Experience replay (failure-weighted sampling)
- Ranking loss on weight predictions
- Direction similarity loss on modification proposals

### 6. Experience Lineage (§30)
Each failure gets a unique experience_id. Future modifications that used it record source_experience_ids, enabling:
- Failure #17 → Memory → Round 132 → Modification → SUCCESS

## Experiment Matrix

| Condition | Memory | Correction | Replay | Direction | Outcome | Rounds |
|-----------|--------|------------|--------|-----------|---------|--------|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | 450 |
| NoMemory | ✗ | ✓ | ✓ | ✓ | ✓ | 450 |
| NoReplay | ✓ | ✓ | ✗ | ✓ | ✓ | 450 |
| NoDirection | ✓ | ✓ | ✓ | ✗ | ✓ | 450 |
| NoOutcome | ✓ | ✓ | ✓ | ✓ | ✗ | 450 |
| PureReversal | ✓ | reversal | ✓ | ✓ | ✓ | 450 |
| ErrorOnly | error-only | ✓ | ✓ | ✓ | ✓ | 450 |
| RandomMemory | random | ✓ | ✓ | ✓ | ✓ | 450 |
| ZeroMemory | zero | ✓ | ✓ | ✓ | ✓ | 450 |

Each condition: 5 seeds × 450 rounds = 2,250 rounds total.

## Phase Analysis (§25)

| Phase | Rounds | Focus |
|-------|--------|-------|
| Early Adaptation | R0-50 | System stabilizes |
| Experience Accumulation | R51-150 | Memory builds up |
| Experience Reuse | R151-300 | Past experiences influence modifications |
| Long-term Stability | R301-450 | Sustained behavior change |

## Metrics

### Primary Acceptance Criteria (§23)

| # | Metric | Target |
|---|--------|--------|
| 1 | SRR_Full > SRR_NoMemory | ✓ (proven in v0.5.1) |
| 2 | RFR_similar,Full < RFR_similar,NoMemory | Must demonstrate |
| 3 | w_success > w_failure | Must demonstrate |
| 4 | P(FutureSimilarFailure) < P(InitialFailure) | Must demonstrate |
| 5 | P(Reuse|Success) > P(Reuse|Failure) | Must demonstrate |
| 6 | Consistent across majority of seeds | Must demonstrate |

### New Metrics (§22)

- EAR = 1 - RFR_future/RFR_baseline (Experience Absorption Rate)
- Modification Direction Similarity
- Target Shift (how target selection changed after experience)
- Magnitude Shift (how magnitude changed after experience)
- Memory Retrieval Similarity (mean cosine of retrieved episodes)

## New Modules

| Module | File | Purpose |
|--------|------|---------|
| ExperienceTracker | experience_absorption.py | Track experiences and lineage |
| ExperienceLineage | experience_absorption.py | Link failures to future successes |
| AbsorptionEvaluator | experience_absorption.py | Compute EAR and absorption metrics |
| FutureModificationEvaluator | future_behavior.py | Track future behavior changes |
| SuccessReuseEvaluator | future_behavior.py | Track successful modification reuse |
| WeightLearner | weight_learning.py | Learn outcome-conditioned weights |
| WeightRankingLoss | weight_learning.py | Enforce w_success > w_failure |
| DirectionEncoder | memory_encoder.py | Encode modification direction |
| OutcomeAwareAttention | memory_encoder.py | Attention with outcome embedding |

## Regression Tests (§36-37)

- Self-modification occurs every round (Δθ ≠ 0)
- Topology unchanged (parameter shapes, layer count, module count)
- Weight bounded in [w_min, w_max]
- Ranking loss correct direction
- Random memory ≠ Zero memory
- Direction similarity works

## Success Criteria

### Demonstrated (v0.5.1)
- ✅ Memory improves SRR

### Must Demonstrate (v0.5.2)
- RFR_similar decreases with experience
- w_success > w_failure consistently
- Outcome-aware memory retrieval produces different behavior
- Direction encoding distinguishes +ΔW from -ΔW
- Success reuse rate > failure reuse rate

### Honest Negative Result
If RFR_Full ≈ RFR_NoMemory after 450 rounds:
> "DSCNS can perform local error correction but has not demonstrated persistent error-experience absorption. Mechanism failure analysis indicates [specific finding]."

## Reproducibility

- Commit hash for each experiment
- Config hash
- Seed-level results (not just mean)
- 4-phase breakdown
- All raw data preserved
