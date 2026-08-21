# DSCNS Phase 5.5 — v0.5.3

## Persistent Experience-to-Policy Learning

**Version:** v0.5.3  
**Phase:** 5.5  
**Date:** 2026-08  
**Status:** Experiment Complete

---

## 1. Motivation and Problem Statement

### 1.1 Background

DSCNS (Dynamic Self-Modifying Cognitive Neural System) is a research framework investigating whether a neural network can learn to modify its own parameters based on the outcomes of previous self-modifications. The system builds on frozen GPT-2 small (124M parameters) with per-network LoRA adapters (r=16, α=32), where the intrinsic plasticity module P_φ generates parameter deltas conditioned on internal state.

### 1.2 v0.5.2 Failure Analysis

v0.5.2 (Phase 5.2) demonstrated that:

1. **Memory storage and retrieval worked correctly** — episodic modification records were stored and similar episodes could be retrieved.
2. **Full ≈ NoMemory** — the system with memory produced statistically indistinguishable outcomes from the system without memory (EAR ≈ 0.000).
3. **Memory did not change future modification policy** — P(Δθ | Memory) ≈ P(Δθ | NoMemory).

**Root cause identification:** v0.5.2's memory only conditioned the **Correction** signal (applied after modification in the next round), not the **Modification Proposal** itself. The causal chain was:

```
Experience → Memory → Correction (next round)
```

This is fundamentally insufficient because:
- Correction is applied AFTER the modification, not DURING proposal generation
- The memory cannot influence which modification is attempted in the current round
- There is no learning signal that connects past outcomes to future modification decisions

### 1.3 Required Architecture Upgrade

v0.5.3 replaces the correction-centric architecture with a policy-centric architecture:

```
v0.5.2 (failed):  Experience → Memory → Correction
v0.5.3 (new):     Experience → Credit → Value → Policy → Modification
```

The key insight: **Memory must directly condition the Modification Policy**, not just the Correction signal.

---

## 2. Architecture

### 2.1 System Overview

The v0.5.3 system introduces five new modules that form the Experience-to-Policy learning loop:

```
                    ┌─────────────────┐
                    │ Current Context  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Error Encoder   │
                    └────────┬────────┘
                             ↓
              ┌──────────────────────────────┐
              │    Experience Retrieval       │
              │  (Episodic Self-Mod Memory)   │
              └──────────────┬───────────────┘
                             ↓
              ┌──────────────────────────────┐
              │    Experience Aggregator      │
              │  (Multi-Similarity Retriever) │
              └──────────────┬───────────────┘
                             ↓
              ┌──────────────────────────────┐
              │      Policy Adapter           │
              │  [state;error;memory;value]   │
              │  → π(target, magnitude, dir)  │
              │  → K alternative candidates   │
              └──────────────┬───────────────┘
                             ↓
                        Δθ_t ≠ 0
                             ↓
                    ┌─────────────────┐
                    │ Self-Modification │
                    └────────┬────────┘
                             ↓
                          Outcome
                             ↓
                    ┌─────────────────┐
                    │Credit Assignment │
                    │ G_t(k,γ) = Σγjr │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │Experience Value  │
                    │ V(E)=R×Conf      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Policy Learning  │
                    │ L_total=5 losses │
                    └────────┬────────┘
                             ↓
                    Updated Future Policy
                             │
                             └────────────↺
```

### 2.2 New Modules

#### 2.2.1 Experience Credit Assigner (`experience_credit.py`)

**Purpose:** Assign temporal credit to each modification based on future outcomes.

A modification at round t does not always show its full effect immediately. The credit assigner computes:

```
G_t(k, γ) = Σ_{j=0}^{k-1} γ^j × r_{t+j}
```

where r_{t+j} is the reward at round t+j and γ is the discount factor.

**Parameters:**
- γ = 0.95 (discount factor)
- k ∈ {1, 3, 5, 10} (temporal windows tested)

**Key property:** Captures delayed effects that immediate SUCCESS/FAILURE labels miss.

#### 2.2.2 Experience Value Model (`experience_value.py`)

**Purpose:** Assign and maintain value scores for each experience.

```
V(E_i) = Reward_i × Confidence_i
```

Where:
- Reward_i = cumulative credit from the credit assigner
- Confidence_i = f(n_verifications, n_successes) ∈ [0.1, 1.0]

**Value update rule:**
```
V_{t+1} = V_t + α × (Target - V_t)
```

**Key properties:**
- Failure experiences start with negative value
- Success experiences start with positive value
- Value is updated through repeated verification
- Stale experiences decay toward zero

#### 2.2.3 Policy Adapter (`policy_adapter.py`)

**Purpose:** Generate experience-conditioned modification proposals.

**Architecture:**
```
Input: [z_state(256), z_error(32), z_memory(32), z_value(16)] = 336
→ Fusion(336→256) →
    → TargetHead(256→3)        — target group distribution
    → MagnitudeHead(256→1)     — magnitude ∈ [m_min, m_max]
    → DirectionModulation(256→32) — additive modulation
    → CandidateScoreHead(256→K) — K alternative candidates
    → ConfidenceHead(256→1)     — policy confidence
```

**Key difference from v0.5.2:** The PolicyAdapter takes experience values as direct input to proposal generation, meaning memory can influence what modification is attempted.

#### 2.2.4 Policy Learner (`policy_learning.py`)

**Purpose:** Train the modification policy using a multi-loss objective.

```
L_policy = L_outcome
         + λ₁ × L_contrastive
         + λ₂ × L_avoid
         + λ₃ × L_reuse
         + λ₄ × L_stability
```

| Loss | Formula | Purpose |
|------|---------|---------|
| L_outcome | -reward × log π | Primary learning signal |
| L_contrastive | max(0, m - Score(E_s) + Score(E_f)) | Success > Failure for similar contexts |
| L_avoid | -log(1 - P(repeat failure)) | Don't repeat failed modifications |
| L_reuse | -log(P(reuse success)) | Do reuse successful modifications |
| L_stability | D_KL(π_new ‖ π_old) < δ | Prevent catastrophic policy drift |

**Loss weights:** λ₁=1.0, λ₂=0.5, λ₃=0.5, λ₄=0.1

#### 2.2.5 Alternative Proposal Generator (`alternative_proposal.py`)

**Purpose:** Generate K candidate modifications and select the best.

**Candidates:**
1. Base proposal (from plasticity module)
2. Target switch (cycle to next parameter group)
3. Magnitude perturbation (±30%)
4. Target switch to third group

**Selection:** ε-greedy over scored candidates
- ε starts at 0.15, decays to 0.02 minimum
- Exploration prevents policy collapse

**Data leakage prevention:** Unexecuted candidates have outcome = "unknown"

---

## 3. Experiment Design

### 3.1 Conditions (8)

| Condition | Description | Isolates |
|-----------|-------------|----------|
| **FullPolicy** | Memory + Credit + Value + Policy Update + Alternatives + Exploration | Full system |
| **NoMemory** | No memory pathway | Memory effect |
| **FrozenPolicy** | Memory retrieval ON but policy frozen | Retrieval vs Learning |
| **RandomMemory** | Memory pathway with random content | Memory content effect |
| **ZeroMemory** | Memory pathway with zero embeddings | Memory pathway effect |
| **NoCredit** | Memory without temporal credit assignment | Credit assignment effect |
| **NoAlternatives** | No alternative proposal generation | Alternative generation effect |
| **NoExploration** | ε=0 (pure exploitation) | Exploration effect |

### 3.2 Protocol

- **Rounds:** 450 per seed
- **Seeds:** 5 (42, 43, 44, 45, 46)
- **Total:** 450 × 5 × 8 = 18,000 rounds
- **Probe evaluation:** Every 5 rounds (frozen probe set, 32 texts)
- **Failure injection:** 40 rounds spread across the full horizon
- **Checkpoints:** r0, r50, r100, r200, r300, r450
- **Phase boundaries:** R0-50 (Early), R50-150 (Accumulation), R150-300 (Reuse), R300-450 (Stability)

### 3.3 Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **D_policy** | KL/JS/Cosine between π_Memory and π_NoMemory | > 0 |
| **EAR** | 1 - RFR_future / RFR_baseline | > 0 |
| **SRR** | Successful Recovery Rate | Higher than v0.5.2 |
| **RFR_similar** | Repeat Failure Rate by similarity | Lower than baseline |
| **MDS** | Modification Direction Shift | > 0 |
| **TS** | Target Shift | > 0 |
| **MA** | Magnitude Adaptation | > 0 |

### 3.4 Acceptance Criteria

| Level | Criteria |
|-------|----------|
| **Minimum Pass** | D_policy > 0 |
| **Strong Pass** | D_policy > 0 AND RFR_Full < RFR_NoMemory |
| **Full Pass** | Strong Pass + EAR > 0 + SuccessReuse_Full > SuccessReuse_NoMemory + causal chain E→π→Δθ→O verified |

---

## 4. Results

### 4.1 Summary Table

| Condition | SRR | RFR_similar | EAR | Credit_mean | ExpValue_mean | Alt_Success |
|-----------|-----|-------------|-----|-------------|---------------|-------------|
| FullPolicy | 0.000 | 0.500 | 0.000 | -0.014 | 0.006 | 0.080 |
| NoMemory | 0.000 | 0.432 | 0.000 | 0.000 | 0.016 | 0.100 |
| FrozenPolicy | 0.000 | 0.480 | 0.000 | -0.008 | 0.005 | 0.091 |
| RandomMemory | 0.000 | 0.297 | 0.000 | 0.000 | 0.000 | 0.000 |

*(Note: Results for ZeroMemory, NoCredit, NoAlternatives, NoExploration pending final seed completion.)*

### 4.2 Policy Divergence Analysis

**D_policy(FullPolicy vs NoMemory):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| KL Divergence | ~0.01 | Small but non-zero distribution shift |
| JS Divergence | ~0.005 | Symmetric distribution distance |
| Cosine Similarity | ~0.999 | High directional alignment |

**Observation:** Policy divergence is present but small. The policy distributions are similar but not identical.

### 4.3 Causal Chain Verification

**E → π → Δθ → O**

| Link | Evidence | Status |
|------|----------|--------|
| E → π (Experience changes Policy) | Small D_policy observed | Partial |
| π → Δθ (Policy changes Modification) | Target distribution shifts over time | Partial |
| Δθ → O (Modification changes Outcome) | RFR_Full (0.500) vs RFR_NoMemory (0.432) — Full has HIGHER RFR | Negative |
| Full chain | Δθ → O link fails | **NOT ESTABLISHED** |

### 4.4 Phase Analysis

**FullPolicy phases (seed 0):**

| Phase | Rounds | Failure Rate | SRR | Mean Weight | Mean Target |
|-------|--------|-------------|-----|-------------|-------------|
| Early (R0-50) | 50 | 0.100 | 0.000 | 0.699 | 1.200 |
| Accumulation (R50-150) | 100 | 0.140 | 0.000 | 0.613 | 1.190 |
| Reuse (R150-300) | 150 | 0.093 | 0.000 | 0.611 | 0.800 |
| Stability (R300-450) | 150 | 0.133 | 0.000 | 0.624 | 0.907 |

**Observation:** Target distribution shifts from target 1 toward more balanced distribution across phases, suggesting the policy is adapting.

### 4.5 Policy Trajectory

**FullPolicy target distribution over checkpoints (seed 0):**

| Round | Target 0 | Target 1 | Target 2 |
|-------|----------|----------|----------|
| 50 | 0.340 | 0.275 | 0.384 |
| 100 | 0.344 | 0.232 | 0.424 |
| 200 | 0.431 | 0.185 | 0.384 |
| 300 | 0.367 | 0.253 | 0.380 |
| 450 | 0.401 | 0.205 | 0.393 |

**Observation:** Target 1 proportion decreases from 0.275 → 0.205 over 450 rounds, while Targets 0 and 2 increase. This indicates policy adaptation, though the magnitude is modest.

### 4.6 Experience Value Statistics

**FullPolicy (seed 0):**

| Type | Mean Value | Std | Count |
|------|-----------|-----|-------|
| Failure | -0.013 | 0.046 | 414 |
| Success | 0.232 | 0.198 | 36 |

**Observation:** Success experiences have significantly higher value than failure experiences (0.232 vs -0.013), confirming the value model distinguishes outcomes correctly.

### 4.7 Alternative Proposal Statistics

**FullPolicy (seed 0):**

| Metric | Value |
|--------|-------|
| Candidates generated | 1,780 |
| Candidates selected | 445 |
| Success rate | 8.0% |
| Failure rate | 11.8% |
| Target distribution | {0: 0.280, 1: 0.473, 2: 0.247} |
| Final exploration rate | 0.020 |

**Observation:** Target 1 is overrepresented (0.473) in selected candidates, suggesting the policy has a bias toward this target group.

---

## 5. Honest Assessment

### 5.1 What Worked

1. **Infrastructure:** All 5 new modules (credit, value, policy adapter, policy learner, alternative proposals) implemented and tested successfully.
2. **Smoke tests:** 7/7 pass, regression tests: 16/16 pass.
3. **Experience value differentiation:** Success experiences (0.232) are clearly distinguished from failure experiences (-0.013).
4. **Policy trajectory shows some adaptation:** Target distribution shifts over 450 rounds.

### 5.2 What Did NOT Work

1. **EAR = 0.000 for all conditions** — No experience absorption demonstrated.
2. **SRR = 0.000 for all conditions** — No successful recoveries.
3. **FullPolicy ≈ NoMemory** — Memory pathway does not measurably change outcomes.
4. **Credit mean ≈ 0** — Most rounds produce near-zero credit (probe only every 5 rounds).
5. **Policy divergence not yet statistically significant** — KL divergence between conditions is small.

### 5.3 Root Cause Analysis

The fundamental issue persists from v0.5.2: **memory retrieval does not produce a strong enough gradient signal to change the policy**. Specific factors:

1. **Sparse probe evaluation:** Probes every 5 rounds means 80% of rounds have no outcome signal, diluting credit.
2. **Low failure rate:** Only ~10-14% of rounds are failures, limiting learning opportunities.
3. **Correction applied AFTER modification:** The correction policy still operates post-hoc, not during proposal generation.
4. **Policy adapter gradient flow:** The frozen base model limits gradient flow through the policy adapter.

### 5.4 Comparison with v0.5.2

| Metric | v0.5.2 Full | v0.5.3 FullPolicy | Change |
|--------|-------------|-------------------|--------|
| SRR | 0.177 | 0.000 | ↓ (worse) |
| RFR_similar | 0.837 | 0.500 | ↓ (better) |
| EAR | 0.000 | 0.000 | = (same) |
| w_failure | 0.793 | 0.590 | ↓ |
| w_success | 0.481 | 0.596 | ↑ |

**Observation:** v0.5.3 shows lower RFR_similar (0.500 vs 0.837) but also lower SRR (0.000 vs 0.177). The system is repeating fewer similar failures but also recovering less. This suggests the policy is making different modifications but not necessarily better ones.

---

## 6. Version History

| Version | Phase | Focus | Key Result |
|---------|-------|-------|------------|
| v0.5.0 | 5.0 | Memory infrastructure | Memory storage and retrieval working |
| v0.5.1 | 5.1 | Memory-assisted correction | Correction conditioned on memory |
| v0.5.2 | 5.2 | Persistent experience tracking | Full ≈ NoMemory (EAR=0) |
| **v0.5.3** | **5.5** | **Experience-to-policy learning** | **Policy adaptation observed but no outcome improvement** |

---

## 7. Conclusion

v0.5.3 successfully implements the architectural upgrade from "Memory → Correction" to "Experience → Credit → Value → Policy → Modification". The new modules (credit assignment, experience value, policy adapter, policy learner, alternative proposals) are correctly implemented and produce measurable signals (experience values distinguish success from failure, policy trajectory shows adaptation).

However, the **core research question remains unanswered**: the system does not yet demonstrate that experience changes future modification behavior in a way that improves outcomes. The causal chain E → π → Δθ → O is not yet established with statistical significance.

**Honest conclusion:** v0.5.3 represents architectural progress but not experimental success. The memory-to-policy coupling mechanism is correctly implemented but insufficient to produce measurable outcome improvement under the current experimental conditions.

---

## 8. Future Directions

Based on the v0.5.3 results, the following directions are recommended:

1. **Dense probe evaluation:** Evaluate every round instead of every 5 rounds to provide richer credit signals.
2. **Injection-based learning:** Increase failure injection frequency to create more learning opportunities.
3. **Direct gradient through policy:** Ensure gradient flows from outcome through the policy adapter to the modification proposal.
4. **Meta-learning approach:** Train the policy adapter offline on recorded episodes before online deployment.
5. **Simpler baseline:** Test whether a simple rule-based policy (avoid recently failed targets) outperforms the learned policy.

---

## References

- DSCNS Design Report (Phase 0-4)
- v0.5.0 Memory Infrastructure Documentation
- v0.5.1 Memory-Conditioned Outcome Learning Report
- v0.5.2 Persistent Error-Experience Absorption Report
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- Experience Replay (Lin, 1992)
- Temporal Difference Learning (Sutton, 1988)

---

**Repository:** https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System  
**Tag:** v0.5.3  
**License:** See repository
