# Known Negative Results

This document records experimental results that did NOT support the original hypothesis. These are scientifically valuable and must be preserved.

## v0.5.2 — Memory ≈ NoMemory

**Hypothesis:** Experience memory should improve self-modification outcomes.

**Result:** RFR_Full ≈ RFR_NoMemory. Memory storage did not produce measurably better modification outcomes.

**Interpretation:** Storing modification history is necessary but not sufficient. The memory was stored and retrieved but never directly influenced the modification policy.

## v0.5.3 — Policy Change → No Outcome Improvement

**Hypothesis:** If memory conditions the modification policy (Experience → Policy), then the conditioned policy should produce better outcomes.

**Result:** D_policy(Full vs NoMemory) = 0.0132 > 0 (policy changed), but RFR_Full (0.459) > RFR_NoMemory (0.438) (outcomes did NOT improve).

**Interpretation:** The policy learning signal was too weak or misdirected. The policy changed, but the change was not outcome-directed. The losses used (contrastive, avoidance, reuse, stability) do not directly tie policy decisions to outcome quality.

## v0.5.3 — NoCredit ≡ FullPolicy

**Hypothesis:** Temporal credit assignment should distinguish policy decisions by their long-term impact.

**Result:** KL(FullPolicy vs NoCredit) = 0.0000, Cosine = 1.0000. The two conditions are identical.

**Interpretation:** Credit assignment had zero effect because probe evaluation occurs only every 5 rounds. 80% of rounds have no outcome signal, making credit signals uniformly zero.

## v0.5.3 — RandomMemory Has Lowest RFR

**Observation:** RandomMemory (RFR=0.297) paradoxically produces fewer repeated failures than FullPolicy (RFR=0.459).

**Interpretation:** Random memory retrieval disrupts systematic repetition of failures. This suggests that "similar" memory retrieval may be retrieving failure-prone contexts, while random retrieval avoids this trap.

## Implications for v0.6.0

These negative results directly motivated v0.6.0's design:

1. **Policy change without outcome improvement** → v0.6.0 adds outcome-directed reward
2. **Credit assignment has zero effect** → v0.6.0 uses denser outcome signals
3. **Random memory outperforms** → v0.6.0 includes Oracle and Random baselines
4. **Policy learning too weak** → v0.6.0 adds direct outcome → policy credit
