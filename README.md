# DSCNS — Dynamic Self-Modifying Cognitive Network System

> **Status: Early Research Prototype — Phase 5 Extended Self-Modifying Loop**

DSCNS is an experimental research prototype exploring whether a neural system
can continuously observe its own state, modify its own parameters, evaluate
the consequences of those modifications, learn from failed modifications, and
repeatedly adapt itself over long horizons.

**Phase 5 currently focuses on intrinsic parameter self-modification rather
than claiming general intelligence or unrestricted autonomous
self-improvement.**

The purpose of Phase 5 is narrow and well-defined:

> Can a neural system repeatedly modify its own parameters through an
> internally generated modification process, observe the consequences, and
> use those consequences to influence future modifications?

- [简体中文](README.zh-CN.md) · [日本語](README.ja-JP.md)

---

## Overview

DSCNS replaces the single "data → gradient update" flow with a closed-loop
cognitive process: experiences are observed by multiple networks, evaluated
independently, verified across networks, and only then progressively
internalized — or stored as callable knowledge instead.

In **Phase 5**, each network hosts an *intrinsic plasticity module* that
produces continuous parameter changes from the network's own internal state.
In **Phase 5.1**, this is extended to mandatory per-round self-modification
with self-determined magnitude, target selection, and error-conditioned
self-correction.

```
Phase 5 Extended Self-Modifying Loop
┌─────────────────────────────────────────────┐
│                                             │
│              Current Parameters θ_t         │
│                       │                     │
│                       ▼                     │
│             Internal State h_t              │
│                       │                     │
│       ┌───────────────┼───────────────┐     │
│       ▼               ▼               ▼     │
│  Target Head    Magnitude Head   Delta Head  │
│       │               │               │     │
│       └───────┬───────┘               │     │
│               ▼                       ▼     │
│     target_group, α_t              Δθ_raw   │
│               │                       │     │
│               └───────────┬───────────┘     │
│                           ▼                 │
│              θ'_{t+1} = θ_t + α_t Δθ_t     │
│                           │                 │
│                           ▼                 │
│                  Fixed Probe evaluation      │
│                           │                 │
│               ┌───────────┴───────────┐     │
│               ▼                       ▼     │
│           Improvement             Failure   │
│               │                       │     │
│               ▼                       ▼     │
│         reinforce useful      Error analysis│
│         modification          + correction  │
│                                    │        │
│                                    ▼        │
│                           modify next Δθ    │
│                                    │        │
└────────────────────────────────────┴────────┘
```

**Key principles:**

| Function | External controller | Model (P_φ) |
|---|---|---|
| Start round / safety / record | ✓ | |
| Decide whether modification occurs | ✓\* | |
| Choose modification target | | ✓ |
| Choose modification direction | | ✓ |
| Choose modification magnitude | | ✓ |
| Learn from failed modification | | ✓ |
| Generate next correction | | ✓ |

\* *External control only guarantees the experimental condition that
modification occurs; it does not determine the modification itself.*

---

## Phase 5 Operational Loop

Each round follows the same autonomous self-modification cycle:

1. **Observe** current internal state (h_t)
2. **Retrieve** previous modification experiences from memory
3. **Generate** a mandatory parameter modification (P_φ)
4. **Predict** modification magnitude (model-determined)
5. **Select** target parameter group (model-determined)
6. **Apply** Δθ via θ' = θ + α_t Δθ_t
7. **Evaluate** on the fixed probe set
8. **Measure** the consequence (success / failure / recovery)
9. **Store** the full (state, proposal, outcome) transition
10. **Learn** from error-conditioned experiences
11. **Continue** to the next round

The external experiment controller only enforces the experimental protocol
and safety constraints. It does not choose whether, where, or how much the
model should modify itself.

---

## Evidence Levels

Results are classified into verifiable evidence levels:

| Level | Proposition | Status |
|---|---|---|
| **L1** | Δθ exists and is non-zero | ✓ Proven |
| **L2** | Δθ is state-dependent | ✓ Proven |
| **L3** | Δθ changes model behavior | ✓ Proven |
| **L4** | Repeated modification produces net parameter drift | ✓ Proven |
| **L5** | Modification magnitude is self-determined | ✓ Proven (P5.1) |
| **L6** | Failed modifications condition future modification behavior | Hypothesis (planned) |
| **L7** | Long-horizon stability over thousands of rounds | ✓ Partially (3000 rounds, regime B) |

---

## Architecture

```
dscns/
├── intrinsic_plasticity.py     # IntrinsicPlasticityModule (P5 + P5.1)
├── error_correction.py         # ErrorEncoder, ErrorState (P5.1)
├── modification_memory.py      # Structural memory + episodic self-mod memory
├── networks.py                 # CognitiveNetwork with self-mod methods
├── plasticity_trainer.py       # P5-C offline training (extended for P5.1)
├── evolution.py                # Structure evolution (P3/P4)
├── self_modification.py        # Learned structural policy (P4)
├── modification_memory.py      # Episodic modification memory (P5.1)
├── config.py                   # Configuration
├── system.py                   # System orchestration
└── ...
```

**Modules not modified in Phase 5.1:** `base_model.py`, `experience.py`,
`internalization.py`, `memory.py`, `metacognition.py`, `communication.py`,
`verification.py`.

---

## Datasets

| Domain | Dataset | Source |
|---|---|---|
| general | Wikitext-103 (Wikipedia) | HuggingFace `wikitext` |
| math | GSM8K | HuggingFace `gsm8k` |
| logic | MATH-500 (fallback) | HuggingFace `HuggingFaceH4/MATH-500` |
| code | HumanEval | HuggingFace `openai/openai_humaneval` |
| science | SciQ | HuggingFace `allenai/sciq` |

---

## Fixed Probe Set

A fixed probe set of 32 texts (5 domains) is generated deterministically
(seed 1234) and frozen before all experiments. Every experimental group
uses the *identical* probe set. The probe set:

- Is never modified, replaced, or selected by the self-modification mechanism
- Does not participate in task training
- Is used every round to measure output drift D_output(t)

This ensures that measured performance changes reflect real behavioral drift
from parameter modification, not changes to the evaluation itself.

---

## Parameter Drift Metrics

P5.1 reports both per-step modification and cumulative drift:

- **Per-step modification:** M_t = ||θ_{t+1} − θ_t||
- **Gross movement:** D_gross(T) = Σ_t ||θ_{t+1} − θ_t||
- **True net drift:** D_net(T) = ||θ_T − θ_0||
- **Drift ratio:** R = D_net / (D_gross + ε) — ≈1 coherent, ≈0 cancelling

---

## Reproduction

Environment: Python 3.8.16, PyTorch 1.13.1, CUDA 11.7
(`transformers==4.45.2`, `peft==0.12.0`; see `requirements.txt`).

```bash
# 1) Download the base model
python scripts/download_model.py

# 2) Prepare datasets
python -c "from scripts.common import make_config, prepare_data; \
           d = prepare_data(make_config()); print({k: len(v) for k, v in d['train'].items()})"

# 3) Phase 1-4 (existing experiments)
python scripts/run_phase1.py --modes control exp1 exp2 --out experiments/phase1
python scripts/run_phase2.py --out experiments/phase2
python scripts/run_phase3.py --out experiments/phase3
python scripts/run_phase4.py --out experiments/phase4

# 4) Phase 5 core validation + long-horizon
python scripts/run_phase5_long_run.py          # 150-round bare P5
python scripts/run_p5_long_horizon.py          # 150-round validation + 3000-round extreme
python scripts/analyze_p5_long_horizon.py

# 5) Phase 5.1: mandatory + magnitude + error correction
python scripts/run_phase5_1.py --arms no_mod,p5_m,p5_mm,p5_mme,random
python scripts/run_phase5_1.py --arms p5_mme --rounds 3000  # extreme horizon

# 6) Analysis + report
python scripts/analyze_phase5_1.py
python scripts/make_report.py
```

---

## Experiment Results

### Phase 1 — continual learning

| Metric | Control | Exp1 | Exp2 |
|---|---|---|---|
| AF ↓ | 0.0037 | **0.0010** | 0.0013 |
| FWT ↑ | 0.0002 | −0.0041 | **+0.0013** |
| CLS ↑ | **0.0868** | 0.0735 | 0.0565 |

### Phase 4 — learned structural self-adaptation

| Metric | fixed | rule | learned |
|---|---|---|---|
| Final mean performance | 0.0554 | 0.0570 | **0.0585** |
| AF ↓ | **0.0000** | 0.0099 | 0.0029 |
| CLS ↑ | 0.0553 | 0.0471 | **0.0556** |

### Phase 5 — intrinsic parameter self-modification

| Metric | fixed | p5_150 | random_control | no_mod |
|---|---|---|---|---|
| Net drift ‖θ₁₅₀−θ₀‖ | 42.46 | 42.46 | 3.78 | 0.00 |
| Gross movement | 46.36 | 46.36 | 46.36 | 0.00 |
| Drift ratio | 0.916 | 0.916 | 0.082 | 0.000 |
| Probe drift | 0.437 | 0.437 | 0.036 | 0.000 |
| Hash constant | — | No | No | **Yes** |

### Phase 5 — 3000-round extreme run

| Metric | Value |
|---|---|
| Net drift | 945.4 |
| Gross movement | 952.2 |
| Drift ratio | **0.993** |
| ‖θ‖ growth | 13.85 → 194.1 |
| Δθ magnitude | Constant ~1.32 (3000 rounds) |
| NaN / Inf | 0 / 0 |
| Regime | **B: continuous drift** |

### Phase 5.1 — mandatory self-modification + magnitude

| Arm | Drift | Probe | Success | Recovery |
|---|---|---|---|---|
| p5_mm (magnitude) | 0.017 | 0.000 | 100% | 0 |
| p5_mme (+ error) | 0.018 | 0.000 | 100% | 0 |
| random (budget-matched) | 1.375 | 0.015 | 100% | 0 |

---

## What DSCNS Does Not Claim

DSCNS does **not** currently claim:

- AGI or general autonomous intelligence
- Human-like cognition or biological equivalence
- Unrestricted self-improvement or recursive intelligence explosion
- Guaranteed performance improvement
- Generalization beyond the tested model and datasets
- That the system "understands" why it modifies itself

The purpose of Phase 5 is narrower:
*Can a neural system repeatedly modify its own parameters through an
internally generated modification process, observe the consequences, and
use those consequences to influence future modifications?*

---

## Current Findings (Established)

1. **Finding 1** — Multi-network collaboration does *not* automatically
   improve absolute performance under a fixed computation budget.
2. **Finding 2** — Progressive internalization appears *promising for
   reducing catastrophic forgetting*.
3. **Finding 3** — Information-gain-based experience selection shows
   *promising results* compared with random selection.
4. **Finding 4** — Dynamic topology evolution is currently *not sufficient
   to outperform a fixed topology*.
5. **Finding 5 (P4)** — A small policy trained by rule-imitation +
   REINFORCE *can* produce structural modification decisions.
6. **Finding 6 (P5)** — Internal state can directly produce state-dependent,
   stable, measurable parameter changes (θ → h → Δθ → θ').
7. **Finding 7 (P5)** — At matched gross movement, P5 produces 11× the net
   drift and 12× the behavioral change of random directions.
8. **Finding 8 (P5)** — Over 3000 rounds, Δθ magnitude stays constant
   (~1.32); parameter drift is linear; no NaN/Inf; regime B (sustained
   drift, no divergence or collapse).
9. **Finding 9 (P5.1)** — Mandatory modification can be sustained with
   self-determined magnitude (m_t ≈ 0.05) and target selection.

---

## Future Work

```
P5    Intrinsic parameter modification (θ → h → Δθ → θ')      ✓
P5.1  Mandatory modification + self-magnitude + error learning   Current
P5.2  Adaptive modification (Level 4: learn when/how/why)       Planned
P5.3  Reusable correction experience library                    Planned
P6    Intrinsic structural self-modification                     Planned
```

---

## Limitations

- Single seed (42); single network (N1); single model (GPT-2 small)
- P5 Δθ generation is off the gradient path (no_grad); P5.1 error
  learning is online but simplified (target/magnitude heads, not full P_φ)
- Fixed external trigger for P5; mandatory trigger for P5.1
- Small probe set (32 texts); no multi-seed statistics
- Performance is descriptive, not probative
- See `docs/LIMITATIONS.md` for the complete list

---

## Citation

```bibtex
@misc{dscns2026,
  title  = {DSCNS: Dynamic Self-Modifying Cognitive Neural System},
  author = {Mousennnn},
  year   = {2026},
  note   = {Version v0.4.0, early research prototype},
  howpublished = {GitHub repository},
  url    = {https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System}
}
```

## License

- **Code:** [GNU General Public License v3.0](LICENSE)
- **Documentation:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Attribution:** When reusing or adapting the documentation, please credit:

> DSCNS — Dynamic Self-Modifying Cognitive Network System (v0.4.0), by
> Mousennnn, licensed under CC BY 4.0.
> https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System
