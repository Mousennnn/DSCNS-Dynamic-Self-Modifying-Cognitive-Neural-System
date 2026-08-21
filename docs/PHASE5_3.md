# DSCNS Phase 5.5 — v0.5.3

## Persistent Experience-to-Policy Learning

### Core Research Question

**Can the model change its future modification strategy based on past self-modification outcomes?**

### Motivation: v0.5.2 Failure Analysis

v0.5.2 demonstrated that:
- Memory storage and retrieval worked
- But **Full ≈ NoMemory** (EAR ≈ 0)
- Memory did not change future modification policy

**Root cause:** v0.5.2's memory only conditioned the **Correction** signal (applied next round), not the **Modification Proposal** itself.

### Architecture Upgrade

**v0.5.2 (failed):**
```
Experience → Memory → Correction
```

**v0.5.3 (new):**
```
Experience → Credit → Value → Policy → Modification
```

### New Modules

| Module | File | Purpose |
|--------|------|---------|
| `ExperienceCreditAssigner` | `experience_credit.py` | Temporal credit assignment with discount γ |
| `ExperienceValueModel` | `experience_value.py` | V(E) = Reward × Confidence, with update |
| `PolicyAdapter` | `policy_adapter.py` | Experience-conditioned modification policy |
| `ModificationPolicyLearner` | `policy_learning.py` | Multi-loss training (contrastive/avoid/reuse/stability) |
| `AlternativeProposalGenerator` | `alternative_proposal.py` | K candidate modifications, ε-greedy selection |

### Key Design Principles

1. **Memory → Policy (not just Correction):** PolicyAdapter takes experience value as direct input to proposal generation
2. **Temporal Credit:** G_t(k,γ) = Σ γ^j r_{t+j} captures delayed effects
3. **Experience Value:** V(E) = reward × confidence, with repeated verification
4. **Alternative Proposals:** K candidates scored by policy, ε-greedy selection
5. **Policy Learning:** 5-loss objective prevents collapse and ensures proper credit assignment

### Experiment Conditions (8)

| Condition | Description |
|-----------|-------------|
| **FullPolicy** | Memory + Credit + Value + Policy Update + Alternatives + Exploration |
| **NoMemory** | No memory pathway (baseline) |
| **FrozenPolicy** | Memory retrieval ON but policy frozen (ablation) |
| **RandomMemory** | Memory pathway with random content |
| **ZeroMemory** | Memory pathway with zero embeddings |
| **NoCredit** | Memory without temporal credit assignment |
| **NoAlternatives** | No alternative proposal generation |
| **NoExploration** | ε=0 (pure exploitation) |

### Evaluation Metrics

- **D_policy** (KL/JS/Cosine between π_Memory and π_NoMemory)
- **Experience Influence** (EI)
- **Failure Avoidance** (FA)
- **Success Reuse** (SRU)
- **EAR** (Experience Absorption Rate)
- **MDS** (Modification Direction Shift)
- **TS** (Target Shift)
- **MA** (Magnitude Adaptation)

### Causal Chain Verification

E → π → Δθ → O (4-level chain)

1. **E → π:** Memory changes policy distribution (D_policy > 0)
2. **π → Δθ:** Policy change alters modification behavior
3. **Δθ → O:** Modification change improves outcomes (RFR decreases)
4. **O → E:** Outcomes create new experiences (loop)

### Acceptance Criteria

- **Minimum Pass:** D_policy > 0
- **Strong Pass:** D_policy > 0 AND RFR_Full < RFR_NoMemory
- **Full Pass:** All above + EAR > 0 + causal chain verified

### Running

```bash
# Smoke test (no GPU needed)
python smoke_v053.py

# Full experiment (450 rounds × 5 seeds × 8 conditions)
python scripts/run_v053.py

# Analysis
python scripts/analyze_v053.py
```

### Version History

- v0.5.0: Memory infrastructure
- v0.5.1: Memory-assisted correction
- v0.5.2: Persistent experience tracking (demonstrated memory→correction gap)
- **v0.5.3: Experience-to-policy learning (new)**
