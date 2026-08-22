# DSCNS Phase 6 / v0.6.0

## Self-Modification Policy Causality & Long-Horizon Relay Learning

### 自修改策略因果诊断与长程持续适应系统

---

## 1. Motivation

v0.5.3 established measurable Experience → Policy coupling (D_policy > 0), but policy change did NOT produce better modification outcomes (RFR_Full > RFR_NoMemory, EAR ≈ 0).

**v0.6.0 asks:** Why has the policy changed without producing better self-modification outcomes?

## 2. v0.5.3 Findings

| Finding | Status |
|---|---|
| Experience → Policy coupling | **Supported** (KL = 0.0132) |
| Policy → Modification causality | **Partial** (target accuracy measured) |
| Modification → Outcome improvement | **Not established** (RFR_Full ≥ RFR_NoMemory) |
| Outcome → Credit → Policy | **Not established** (EAR = 0) |
| Complete closed loop | **Not established** |

## 3. v0.6.0 Architecture

```
                         ┌───────────────────┐
                         │ Current Context   │
                         └─────────┬─────────┘
                                   ↓
                         ┌───────────────────┐
                         │ Error Encoder     │
                         └─────────┬─────────┘
                                   ↓
             ┌────────────────────────────────────┐
             │ Experience Retrieval               │
             └────────────────┬───────────────────┘
                              ↓
                    ┌───────────────────┐
                    │ Experience        │
                    │ Aggregator        │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Modification      │ ← Outcome-directed reward
                    │ Policy            │
                    └─────────┬─────────┘
                              ↓
               ┌────────────────────────────┐
               │ Candidate Generator       │
               │ A1 / A2 / ... / AK         │
               └─────────────┬──────────────┘
                             ↓
                     Policy Selection (adaptive ε)
                             ↓
                       Safety Envelope
                             ↓
                       Proposal a_t
                             ↓
                  ┌─────────────────────┐
                  │ Apply Δθ            │
                  └──────────┬──────────┘
                             ↓
                         Outcome
                             ↓
               ┌────────────────────────────┐
               │ Outcome-Directed Reward    │
               │ R_t = w_p·R_p + w_e·R_e   │
               │       + w_s·R_s + w_c·R_c  │
               └─────────────┬──────────────┘
                             ↓
                    Credit Assignment
                             ↓
                     Policy Update
                             ↺
```

## 4. New Modules (v0.6.0)

| Module | Purpose | Input | Output |
|---|---|---|---|
| `policy_trace.py` | Policy-to-modification tracing | Round state | TraceEntry |
| `outcome_policy_learning.py` | Outcome-directed reward + credit | Performance, params | Reward, Credit |
| `modification_guard.py` | Safety envelope for drift | Norms, entropy, KL | Magnitude scale |
| `checkpoint_manager.py` | Best/Final/Relay checkpoints | State dict | Saved files |
| `relay_manager.py` | Cross-version relay lineage | Relay state | Relay checkpoint |

## 5. Experimental Conditions (11)

| # | Condition | Description |
|---|---|---|
| 1 | **FullPolicy** | Complete v0.6.0 system |
| 2 | **NoMemory** | Memory retrieval disabled |
| 3 | **FrozenPolicy** | Policy not updated |
| 4 | **RandomMemory** | Memory retrieval randomized |
| 5 | **ZeroMemory** | Memory replaced with zeros |
| 6 | **NoCredit** | Temporal credit disabled |
| 7 | **NoAlternatives** | Single candidate (no diversity) |
| 8 | **NoExploration** | ε = 0 (pure exploitation) |
| 9 | **NoOutcomeReward** | Outcome-directed reward disabled |
| 10 | **Oracle** | Oracle policy (upper bound) |
| 11 | **Random** | Random policy (lower bound) |

## 6. Outcome-Directed Reward

```
R_t = w_p · ΔPerformance + w_e · ΔError + w_s · (-Drift) + w_c · Consistency
```

Key: Uses **delta** (not absolute) values:
- `ΔPerformance = Performance(θ_t) - Performance(θ_{t-1})`
- `ΔError = Error(θ_{t-1}) - Error(θ_t)`

## 7. Safety Envelope

Monitors:
- Parameter drift from stable state
- Parameter norm magnitude
- Policy entropy (collapse detection)
- Policy KL (instability detection)
- Probe performance (catastrophic forgetting)

When risk is high: reduces magnitude (NEVER sets to zero).

## 8. Checkpoint System

Three checkpoint types:

| Type | Selection | Purpose |
|---|---|---|
| **Best** | Validation score | Best observed state |
| **Final** | Round 450 | End-of-experiment state |
| **Relay** | Full state | Cross-version continuation |

Best ≠ Final ≠ Relay (three different model states).

## 9. Relay Learning Protocol

```
v0.5.3 → Relay₀ → 450 rounds → Relay₁ → 450 rounds → Relay₂ → ...
```

Each relay saves: model_state, policy_state, optimizer, memory, experience_values, random_state, lineage.

## 10. Evidence Levels (v0.6.0)

| Level | Proposition | Status |
|---|---|---|
| L1 | Non-zero parameter modification | **Validated** |
| L2 | State-dependent modification | **Validated** |
| L3 | Modification changes behavior | **Validated** |
| L4 | Repeated modification → measurable drift | **Validated** |
| L5 | Modification determined by internal mechanism | **Validated** |
| L6 | Experience changes future Policy | **Supported** |
| L7 | Policy changes actual Modification | **Partial** |
| L8 | Modification changes Outcome | **Not established** |
| L9 | Outcome forms effective Credit → Policy | **Not established** |
| L10 | Complete loop repeatable across seeds | **Not established** |
| L11 | Long-term stable self-modification | **Testing** |
| L12 | Cross-version relay effectiveness | **Testing** |

## 11. Acceptance Criteria

| Level | Criterion | Description |
|---|---|---|
| **Minimum Pass** | Locate one failure mechanism | Controlled ablation validates a specific cause |
| **Mechanism Pass** | Policy → Actual Modification correlation | target_accuracy > chance AND magnitude_correlation > 0 |
| **Strong Pass** | Outcome-directed improvement | D_policy > 0 AND RFR_Full < RFR_NoMemory |
| **Full Pass** | Complete closed loop | Strong Pass + EAR > 0 |
| **Milestone Pass** | Long-horizon stability | Relay experiment shows no collapse |

## 12. Known Limitations

- GPT-2 small (124M) — limited model capacity
- Single-seed deterministic probe (32 samples, 5 domains)
- Fixed external trigger (probe every 5 rounds)
- No gradient through modification path
- Synthetic/simple task suite
- Policy learning operates on limited error representation
- Long-horizon relay may accumulate compounding errors

## 13. Reproducibility

| Item | Value |
|---|---|
| Python | 3.8.16 |
| PyTorch | 1.13.1+cu117 |
| CUDA | 11.7 |
| Transformers | 4.45.2 |
| PEFT | 0.12.0 |
| GPU | RTX 3070 Ti 8GB |
| Seeds | 42, 43, 44, 45, 46 |
| Rounds | 450 per seed |
| Config | config/phase6.yaml |

## 14. Usage

### Run experiments
```bash
python scripts/run_phase6.py --rounds 450 --seeds 5
```

### Run analysis
```bash
python scripts/analyze_phase6.py --dir experiments/phase6
```

### Generate figures
```bash
python scripts/plotting/plot_phase6.py --input experiments/phase6
```

### Run inference
```bash
python scripts/infer.py --mode self-modify --checkpoint experiments/phase6/checkpoints/FullPolicy/seed_0/best/model.pt
```

### Run demo
```bash
python scripts/demo_inference.py
```

### Run smoke test
```bash
python scripts/run_phase6.py --smoke
```
