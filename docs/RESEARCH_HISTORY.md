# DSCNS Research History

## Version Timeline

| Version | Phase | Milestone | Key Finding |
|---|---|---|---|
| v0.1.0 | P1 | Multi-Network Architecture | 5 cognitive networks with domain specialization |
| v0.2.0 | P2 | Multi-Network Communication | Inter-network message passing and knowledge sharing |
| v0.3.0 | P3 | Structure Evolution | Network split/merge based on domain overlap |
| v0.4.0 | P4 | Learned Structural Adaptation | Neural controller for topology decisions |
| v0.5.0 | P5 | Intrinsic Parameter Self-Modification | GPT-2 can modify its own LoRA parameters via P_phi |
| v0.5.1 | P5.1 | Error-Conditioned Correction | Mandatory self-modification with magnitude control |
| v0.5.2 | P5.2 | Persistent Experience | Episodic memory of modification history |
| v0.5.3 | P5.3 | Experience-to-Policy | Memory directly conditions modification policy |
| v0.6.0 | P6 | Policy Causality + Outcome-directed Learning | Causal diagnosis framework + Best/Final/Relay checkpoints |

## Evidence Chain

### v0.5.0 — Parameter Self-Modification
- θ → h → Δθ → θ' works (parameter modification produces measurable change)
- **Validated:** L1 (non-zero modification)

### v0.5.1 — Error-Conditioned Modification
- Modification magnitude and target determined by error state
- **Validated:** L2 (state-dependent), L5 (magnitude/target by internal state)

### v0.5.2 — Experience Memory
- Modification history stored in episodic memory
- Memory ≈ NoMemory for outcome (honest negative result)
- **Validated:** L3 (behavior change), L4 (measurable drift)
- **Not established:** Memory doesn't improve outcomes

### v0.5.3 — Experience-to-Policy
- Memory directly conditions modification policy
- D_policy(Full vs NoMemory) = 0.0132 > 0
- **Supported:** L6 (experience changes policy)
- **Not established:** L7-L10 (policy change doesn't improve outcomes)

### v0.6.0 — Policy Causality + Outcome-directed
- Causal diagnosis framework (11 conditions)
- Outcome-directed reward (delta-based)
- Safety envelope (risk-based magnitude scaling)
- Best/Final/Relay checkpoint system
- **Testing:** L7-L12

## Known Negative Results

| Version | Finding | Implication |
|---|---|---|
| v0.5.2 | Memory ≈ NoMemory | Memory alone doesn't improve outcomes |
| v0.5.3 | Policy change → no outcome improvement | Policy learning signal too weak |
| v0.5.3 | NoCredit ≡ FullPolicy | Credit assignment has zero effect |

## Open Questions

1. Why does policy change not produce better outcomes?
2. Is the error representation sufficient for meaningful policy learning?
3. Can outcome-directed reward close the causal loop?
4. Does long-horizon relay maintain stable self-modification?
