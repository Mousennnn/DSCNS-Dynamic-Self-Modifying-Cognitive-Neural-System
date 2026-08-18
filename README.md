# DSCNS — Dynamic Self-Modifying Cognitive Network System

> **Status: Early Research Prototype (v0.1.0)**

DSCNS is an experimental research prototype exploring a dynamic modular
neural architecture for **continual learning**, **candidate knowledge
validation**, **selective internalization**, **memory**, **meta-cognition**,
and **structural evolution**.

The system is designed around the following conceptual loop:

```
Experience
  → Multi-network observation
  → Independent evaluation
  → Cross-network verification
  → Meta-cognitive decision
  → Selective progressive internalization
  → Regression testing
  → Memory update
  → Structural adaptation
  → Continual learning
```

This repository contains the current prototype implementation and its
experimental results (Phases 0–3). It is intended as an open, honest,
reproducible research artifact — **not** as a claim of solved continual
learning, AGI, or human-level intelligence.

- [简体中文](README.zh-CN.md) · [日本語](README.ja-JP.md)

---

## Overview

DSCNS replaces the single "data → gradient update" flow with a closed-loop
cognitive process: experiences are observed by multiple networks, evaluated
independently, verified across networks, and only then progressively
internalized — or stored as callable knowledge instead. The network topology
itself is treated as a learnable structure (split / merge / connect).

The current prototype builds on a **frozen GPT-2 small (124M) base model**
with **one LoRA adapter per cognitive network**, and implements the full
verification / internalization / memory / meta-cognition / evolution stack
described in the design document.

## Core Idea

| Traditional paradigm | DSCNS paradigm |
|---|---|
| data → model → gradient descent → fixed parameters | experience → multi-network observation → independent evaluation → cross-network verification → selective internalization → structural reorganization → continual evolution |

Ten design principles (abridged): receiving information ≠ learning;
learning ≠ immediate parameter modification; parameter updates must be
verified; the same experience can be observed by multiple networks;
observing ≠ internalizing; knowledge can be shared or locally internalized;
forgetting should be local, gradual, and reactivatable; networks can
communicate, correct each other, and form new connections; the network
structure itself is a learning outcome; the system continuously changes
itself through experience.

## System Architecture

```
        External environment / experience stream
                            │
              Experience Management Layer
        (Experience Buffer → Candidate Parser → Active Selector)
                            │ parsed candidates
                            ▼
            Multi-Network Cognitive Layer (N1 … N5)
        (shared frozen base model + independent LoRA adapters)
                            │ evaluations Q_i = (R, N, C, I)
                            ▼
          Cross-Network Verification Layer
        (trust-weighted aggregation · conflict detection/resolution)
                            │ verified knowledge
                            ▼
             Meta-Cognitive Control Layer
        (update decisions · adaptive learning rate · structure evolution)
                            │
                            ▼
             Memory & Knowledge Store
        (Episodic · Semantic · Procedural)
```

## Key Mechanisms

### Multi-network representation

Multiple cognitive networks share one frozen base model but hold **independent
LoRA adapters** — i.e. a distributed parameter space

```
Θ_t = {θ_1, θ_2, …, θ_k}
```

independent per network while the base weights stay shared (memory-efficient).

### Candidate knowledge

New experiences do **not** modify parameters directly. They first become
*candidate knowledge* and pass through evaluation, verification, and an
internalization decision.

### Independent evaluation

Each network scores a candidate along four dimensions:

```
Q_i = (R, N, C, I)
```

- **R** — Relevance (similarity to the network's domain)
- **N** — Novelty (dissimilarity to already-internalized knowledge)
- **C** — Confidence (base-model evidence × source reliability)
- **I** — Importance (relevance × uncertainty-driven utility)

### Cross-network verification

Per-network confidence is aggregated with trust weighting,

```
w_i = trust_i × R_i,        C_final = Σ w_i·C_i / Σ w_i
```

Conflicts (`max(C_i) − min(C_i) > threshold`) trigger evidence-based
resolution; when evidence is insufficient the system **defers** instead of
forcing an answer. Network trust weights are updated dynamically from
observed correctness.

### Progressive internalization

Knowledge enters a network through a gated loop:

```
Tentative update → Regression test → Accept / Rollback
```

with an update-budget constraint

```
‖Δθ‖₂ ≤ ε·‖θ‖₂
```

so that single knowledge items cannot cause violent changes to a network.

### Knowledge levels

The prototype records the internalization level of individual knowledge items:

| Level | State |
|---|---|
| Level 1 | **Broadcast observation** — the system knows the knowledge exists |
| Level 2 | **Callable knowledge** — stored in semantic memory, queryable |
| Level 3 | **Internalized knowledge** — affects the network's parameters |

### Memory system

Three functional memory layers:

- **Episodic memory** — time-ordered raw experiences with retrieval
- **Semantic memory** — a lightweight knowledge-graph representation
- **Procedural memory** — successful action sequences per task type

The architecture is *inspired by the functional distinction* between
episodic, semantic, and procedural memory; it does **not** claim to model
human brain mechanisms.

## Current Implementation

| Component | Module | Design-report section |
|---|---|---|
| Experience buffer / candidate parser | `dscns/experience.py` | §2.1, §3.1 |
| Cognitive networks + Q=(R,N,C,I) | `dscns/networks.py` | §2.1, §3.2–3.3 |
| Cross-network verification | `dscns/verification.py` | §3.4, §8.2 |
| Progressive internalization | `dscns/internalization.py` | §3.5, §8.3 |
| Meta-cognitive controller | `dscns/metacognition.py` | §6 |
| Message bus / communication | `dscns/communication.py` | §8.1 |
| Three-layer memory | `dscns/memory.py` | §5 |
| Structural evolution | `dscns/evolution.py` | §4 |
| System orchestration | `dscns/system.py` | §7.3, §11 |
| Metrics (AF/FWT/CLS) | `dscns/evaluation.py` | §7.4, §10.3 |

## Experimental Setup

- **Base model:** GPT-2 small (124M), frozen, local copy
- **Networks:** 5 cognitive networks (N1 world, N2 math, N3 logic,
  N4 language, N5 verification), each with LoRA (r=16) adapters
- **Experience stream:** 24 rounds = general(4) → math(4) → logic(4) →
  code(4) → science(4) → mixed(4), 32 experiences per round
- **Budget parity:** Control / Exp1 / Exp2 all use 8 gradient steps per round
  (Control: unconditional; Exp1/Exp2: regression-gated)
- **Performance metric:** exp(−masked-CE-loss); generation accuracy reported
  separately
- **Hardware:** NVIDIA RTX 3070 Ti 8 GB (original experiments)

## Datasets

| Domain | Dataset | Source |
|---|---|---|
| general | Wikitext-103 (Wikipedia) | HuggingFace `wikitext` |
| math | GSM8K | HuggingFace `gsm8k` |
| logic | MATH-500 (fallback) | HuggingFace `HuggingFaceH4/MATH-500` |
| code | HumanEval | HuggingFace `openai/openai_humaneval` |
| science | SciQ | HuggingFace `allenai/sciq` |

> Note: `hendrycks/competition_math` requires dataset access approval; the
> loader automatically falls back to the public MATH-500.

**Model weights and dataset caches are intentionally excluded from version
control.** They can be downloaded automatically with the provided scripts
(see [Reproduction](#reproduction)).

## Results

Detailed numbers are in `experiments/comparison.md` and `REPORT_zh.md`
(Chinese reproduction report). Summary:

| Phase | Experiment | Result |
|---|---|---|
| Phase 1 | Multi-network continual learning | Mixed — not consistently better than control |
| Phase 1 | Progressive internalization (Exp1/Exp2) | **Reduced forgetting** (AF 0.0037 → 0.0010 / 0.0013) |
| Phase 1 | Forward transfer (Exp2) | Only positive FWT among tested modes (+0.0013) |
| Phase 2 | Information-gain sampling | Best strategy among those tested (0.0591 vs random 0.0572) |
| Phase 3 | Dynamic topology (split/merge/connect) | Mechanism works; currently below fixed topology |

### Phase 1 — continual learning (Control / Exp1 / Exp2)

| Metric | Control | Exp1 | Exp2 |
|---|---|---|---|
| Average Forgetting (AF) ↓ | 0.0037 | **0.0010** | 0.0013 |
| Forward Transfer (FWT) ↑ | 0.0002 | −0.0041 | **+0.0013** |
| Continual Learning Score (CLS) ↑ | **0.0868** | 0.0735 | 0.0565 |
| Mean acquisition ↑ | 0.0942 | 0.0725 | 0.0548 |
| Mean retention ↑ | **0.0905** | 0.0745 | 0.0578 |

*Control = sequential fine-tuning; Exp1 = single network + selective
internalization; Exp2 = 5 networks + cross-network verification.*

![Phase 1 curves](experiments/phase1_comparison_curves.png)
![Phase 1 metrics](experiments/phase1_metrics.png)

### Phase 2 — active learning

| Strategy | Final performance | Final coverage |
|---|---|---|
| random (baseline) | 0.0572 | 0.256 |
| uncertainty | 0.0466 | 0.256 |
| **info_gain** | **0.0591** | 0.256 |
| meta | 0.0561 | 0.256 |

![Phase 2 curves](experiments/phase2_curves.png)

### Phase 3 — structural evolution

| Metric | fixed | evolve |
|---|---|---|
| Final mean performance (5 domains) | **0.0575** | 0.0533 |
| Code-domain adaptation (round 4→7 Δ) | **+0.0130** | +0.0041 |
| Final network count | 5 | 5 |

Evolution events observed: merge (round 7), split (round 9), split (round 11),
merge + dynamic connections (round 13).

![Phase 3 code curve](experiments/phase3_code_curve.png)

## Current Findings

These findings are **preliminary**, limited to this prototype and its
experimental setup, and should not be interpreted as general conclusions
about continual learning or neural architecture design.

1. **Finding 1** — Multi-network collaboration does *not* automatically
   improve absolute performance under a fixed computation budget.
2. **Finding 2** — Progressive internalization appears *promising for
   reducing catastrophic forgetting* in the current prototype
   (AF: 0.0037 → 0.0010 / 0.0013; per-domain forgetting consistently lower).
3. **Finding 3** — Information-gain-based experience selection shows
   *promising results* compared with random selection in the tested setting.
4. **Finding 4** — Dynamic topology evolution is currently *not sufficient
   to outperform a fixed topology* and requires better structural-plasticity
   control (short-term perturbation during split/merge operations was
   observed, consistent with the design report's risk analysis).

## Reproduction

Environment: Python 3.8.16, PyTorch 1.13.1, CUDA 11.7
(`transformers==4.45.2`, `datasets==2.21.0`, `peft==0.12.0`,
`accelerate==0.34.2`, `huggingface_hub==0.25.2`, `tokenizers==0.20.1`;
see `requirements.txt`). The original experiments were conducted on an
NVIDIA RTX 3070 Ti 8 GB.

```bash
# 1) Download the base model (GPT-2 small) — ~550 MB
python scripts/download_model.py

# 2) Prepare datasets (downloads and caches into data/)
python -c "from scripts.common import make_config, prepare_data; \
           d = prepare_data(make_config()); print({k: len(v) for k, v in d['train'].items()})"

# 3) Phase 1 — continual learning (Control / Exp1 / Exp2)
python scripts/run_phase1.py --modes control exp1 exp2 --out experiments/phase1

# 4) Phase 2 — active learning
python scripts/run_phase2.py --out experiments/phase2

# 5) Phase 3 — structural evolution
python scripts/run_phase3.py --out experiments/phase3

# 6) Aggregate results into experiments/comparison.md + plots
python scripts/make_report.py
```

Notes for restricted networks: if the HuggingFace Hub or PyPI are
unreachable from your network, the repository provides a resumable
resolver/downloader (`scripts/resolve_deps.py` + `scripts/download_wheel.py`)
that was used to bootstrap the original environment through a local proxy.

## Project Structure

```
dscns/
├── dscns/                  # core implementation (15 modules)
├── scripts/                # download / experiment / report scripts
├── config/phase1.yaml      # experiment configuration
├── docs/                   # design, experiments, limitations, licenses
├── experiments/            # official results (JSON + plots) — kept in git
├── REPORT_zh.md            # Chinese reproduction report
├── requirements.txt
├── LICENSE                 # GPL-3.0 (code)
└── README(.zh-CN/.ja-JP).md
```

`models/`, `data/`, `wheelhouse/` are local caches and are excluded from
version control by `.gitignore`.

## Design Notes

- The implementation follows the DSCNS design report v1.0
  (see `docs/DESIGN.md` for the English design summary).
- Multi-network = shared frozen base + per-network LoRA adapters, keeping
  the parameter spaces Θ_i independent while sharing memory.
- Knowledge-state levels and internalization degrees I_ij ∈ [0,1] are
  recorded per knowledge item for traceability.
- Known deviations from the design report are documented in
  `docs/EXPERIMENTS.md` (§7) and `docs/LIMITATIONS.md`.

## Limitations

- Small model scale (124M base) and small data budgets.
- Multi-network experiments are bound by a strict computation-budget parity.
- The structural-evolution mechanism is intentionally simple.
- No large-scale benchmarks; no evidence of generalization to other models
  or domains yet.
- No evidence that DSCNS outperforms mature continual-learning methods
  (EWC, experience replay, etc.) in general.
- See `docs/LIMITATIONS.md` for the full list.

## Future Work

- Longer training horizons and larger budgets to re-test hypotheses H1/H2.
- Learned (rather than hand-tuned) relevance and trust-weight functions.
- Better structural-plasticity control (e.g., adaptive evolution thresholds,
  post-evolution stabilization schedules).
- Larger-scale benchmarks (EWC / replay / PEFT baselines).
- Multimodal and open-environment extensions (design report Phases 4–6).

## Citation

If you use this repository in your work, please cite it as:

```bibtex
@misc{dscns2026,
  title  = {DSCNS: Dynamic Self-Modifying Cognitive Network System -- A Research Prototype},
  author = {Mousennnn},
  year   = {2026},
  month  = {aug},
  note   = {Version v0.1.0, early research prototype},
  howpublished = {GitHub repository},
  url    = {https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System}
}
```

## License

- **Code:** [GNU General Public License v3.0](LICENSE)
- **Design documentation & experimental reports** (`docs/`,
  `REPORT_zh.md`, `experiments/comparison.md`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `docs/LICENSE-docs.md`.

**Attribution:** When reusing or adapting the documentation, please credit:

> DSCNS — Dynamic Self-Modifying Cognitive Network System (v0.1.0), by
> Mousennnn, licensed under CC BY 4.0.
> https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System

---

*This is an early research prototype. Experimental results are reported
honestly, including negative or mixed findings.*
