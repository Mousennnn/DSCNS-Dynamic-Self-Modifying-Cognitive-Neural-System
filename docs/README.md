# DSCNS Documentation Index

**Current Stable Release:** v0.6.0
**Current Research Phase:** P6
**Previous Release:** v0.5.3

---

## Core Documentation

| Document | Description |
|---|---|
| [PHASE6.md](PHASE6.md) | Phase 6 specification and architecture |
| [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md) | Complete version timeline and evidence chain |
| [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) | Honest recording of failed hypotheses |
| [DESIGN.md](DESIGN.md) | Original architecture specification |

## Phase Documentation

| Document | Phase |
|---|---|
| [PHASE4.md](PHASE4.md) | Phase 4 — Learned Structural Adaptation |
| [PHASE5.md](PHASE5.md) | Phase 5 — Intrinsic Parameter Self-Modification |
| [PHASE5_1.md](PHASE5_1.md) | Phase 5.1 — Error-Conditioned Correction |
| [PHASE5_2.md](PHASE5_2.md) | Phase 5.2 — Persistent Experience |
| [PHASE5_3.md](PHASE5_3.md) | Phase 5.3 — Experience-to-Policy |
| [PHASE6.md](PHASE6.md) | Phase 6 — Policy Causality & Long-Horizon Learning |

## Experiment Protocols

- Standard: 450 rounds × 5 seeds × N conditions
- Long-Horizon: 450/900/1350/1800 rounds
- Relay: Previous version relay → 450 rounds → New relay

## Reproducibility

| Item | Value |
|---|---|
| Python | 3.8.16 |
| PyTorch | 1.13.1+cu117 |
| CUDA | 11.7 |
| Transformers | 4.45.2 |
| PEFT | 0.12.0 |
| GPU | RTX 3070 Ti 8GB |
