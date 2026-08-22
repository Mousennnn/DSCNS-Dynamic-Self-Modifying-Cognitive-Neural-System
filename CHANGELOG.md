# Changelog

## v0.6.0 — Phase 6: Self-Modification Policy Causality & Long-Horizon Relay Learning

### Major Milestone
This version establishes the experimental and architectural framework for determining whether experience-conditioned self-modification can become outcome-directed and persistent over long horizons.

### New Modules
- `dscns/policy_trace.py` — Policy-to-modification tracing for causal diagnosis
- `dscns/outcome_policy_learning.py` — Outcome-directed reward (R = w_p·ΔPerf + w_e·ΔErr + w_s·Drift + w_c·Consistency)
- `dscns/modification_guard.py` — Safety envelope (risk-based magnitude scaling, never zero)
- `dscns/checkpoint_manager.py` — Best/Final/Relay checkpoint management with SHA256 integrity
- `dscns/relay_manager.py` — Cross-version relay lineage tracking

### New Scripts
- `scripts/run_phase6.py` — 450-round experiments, 11 conditions × 5 seeds
- `scripts/analyze_phase6.py` — Full analysis with evidence matrix
- `scripts/infer.py` — Inference pipeline (baseline/self-modify/relay modes)
- `scripts/demo_inference.py` — Automated demo generation
- `scripts/plotting/plot_phase6.py` — Figure generation from experiment data
- `smoke_v060.py` — Module-level smoke tests
- `tests/test_v060.py` — 69 regression tests (all pass)

### New Config
- `config/phase6.yaml` — Full v0.6.0 experiment configuration (11 conditions)

### Experimental Conditions (11)
1. FullPolicy — Complete system
2. NoMemory — Memory disabled
3. FrozenPolicy — Policy not updated
4. RandomMemory — Random memory retrieval
5. ZeroMemory — Zero memory vectors
6. NoCredit — No temporal credit
7. NoAlternatives — No candidate diversity
8. NoExploration — No exploration (ε=0)
9. NoOutcomeReward — No outcome-directed reward
10. Oracle — Oracle upper bound
11. Random — Random lower bound

### Key Changes from v0.5.3
- Added outcome-directed reward (delta-based, not absolute)
- Added adaptive exploration (ε varies with policy uncertainty)
- Added safety envelope (prevents dangerous parameter drift)
- Added best/final/relay checkpoint system
- Added policy-to-modification trace for causal diagnosis
- Extended from 8 to 11 experimental conditions
- Added Oracle and Random policy baselines

### Scientific Question
Why has the policy changed without producing better self-modification outcomes?

### v0.5.3 → v0.6.0
- v0.5.3: Experience → Policy coupling established (KL=0.0132 > 0)
- v0.5.3: But policy change did NOT improve outcomes (RFR_Full > RFR_NoMemory)
- v0.6.0: Adds causal diagnosis framework + outcome-directed learning

---

## v0.5.3 — Phase 5.5: Experience-to-Policy Learning

### Key Finding
Experience changes the modification policy (D_policy > 0), but policy change does not produce better outcomes.

### Modules Added
- `experience_credit.py` — Temporal credit assignment (γ=0.95)
- `experience_value.py` — Experience value model (V = Reward × Confidence)
- `policy_adapter.py` — Experience-conditioned modification policy
- `policy_learning.py` — Multi-loss policy training
- `alternative_proposal.py` — K-candidate proposal generation

### Experiments
- 450 rounds × 5 seeds × 8 conditions = 18,000 rounds
- Policy divergence: FullPolicy vs NoMemory KL=0.0132

---

## v0.5.2 — Phase 5.2: Persistent Experience

### Key Finding
Experience memory ≈ No memory for outcome (RFR similar).

---

## v0.5.1 — Phase 5.1: Error-Conditioned Correction

### Key Finding
Error-aware self-modification with mandatory magnitude.

---

## v0.5.0 — Phase 5: Intrinsic Parameter Self-Modification

### Key Finding
GPT-2 small can modify its own LoRA parameters guided by internal state.

---

## v0.4.0 — Phase 4: Learned Structural Self-Adaptation

---

## v0.3.0 — Phase 3: Structure Evolution

---

## v0.2.0 — Phase 2: Multi-Network Communication

---

## v0.1.0 — Phase 1: Multi-Network Architecture
