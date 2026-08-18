# DSCNS Limitations

This document states the limitations of the DSCNS research prototype
explicitly and conservatively. It is part of the research record and is
intended to keep the repository scientifically honest.

## 1. Scale and generality

- The base model is small (GPT-2 small, 124M parameters) and the data
  budgets are small (24 rounds × 32 experiences, 4–5 domains).
- **No large-scale benchmarks** have been run (e.g., ImageNet-scale
  continual learning, long-horizon language streams, EWC/replay/PEFT
  baselines at comparable scale).
- **No evidence of generalization** to other model families, languages, or
  domains has been established.
- Results are from a **single GPU setting** (RTX 3070 Ti 8 GB) and may be
  sensitive to hardware, library versions, and seeds.

## 2. Computational constraints

- Multi-network experiments are bound by a strict **computation-budget
  parity** (8 gradient steps per round for every mode). Under this budget,
  multi-network configurations did **not** reach the absolute performance of
  the Control baseline — this is a finding, not a bug.
- Verification costs (cross-domain regression tests per trial) make Exp1/Exp2
  substantially more expensive per round than Control; the design report's
  Risk-1 (verification cost) was observed directly.

## 3. Mechanisms are intentionally simple

- The structural-evolution mechanism (split/merge/connect) is a prototype:
  triggers are hand-tuned thresholds; there is **no learned control policy**.
- Relevance and trust-weight functions are hand-designed heuristics.
- The semantic memory is a lightweight knowledge graph, not a learned
  knowledge base.
- The meta-cognitive layer is a rule-based controller, not a learned one.

## 4. Hypotheses status (as of v0.1.0)

| Hypothesis | Status in this prototype |
|---|---|
| H1 — multi-network verified learning beats EWC/replay/fine-tuning overall | **Not supported** (mixed; absolute performance below Control) |
| H2 — dynamic structure beats fixed structure under shift | **Not supported** (evolution caused short-term perturbation) |
| H3 — experience–parameter decoupling + verification reduces forgetting | **Partially supported** (AF reduced; per-domain forgetting consistently lower) |
| H4 — shared experience beats shared parameters | **Indirect evidence only** (trust weights differentiated; no direct ablation) |

These statuses are limited to the exact settings in `docs/EXPERIMENTS.md`.
They must not be extrapolated to general claims about continual learning or
neural architecture design.

## 5. Known engineering limitations

- Windows path handling: peft's `save_pretrained`/`load_adapter` misread
  local Windows paths as Hub repo ids; the prototype therefore copies
  adapter state **in memory** during splits.
- On this machine, direct TLS to PyPI / HuggingFace Hub was unstable; a
  local proxy and resumable downloads were required. Reproducing on other
  networks may need proxy configuration (see README).
- 8 GB VRAM caps sequence length at 192 tokens and batch size at 8 for this
  stack (torch 1.13 + fp32).
- `hendrycks/competition_math` requires dataset access approval; the loader
  falls back to MATH-500, so logic-domain results are not directly
  comparable to MATH-full benchmarks.

## 6. Claims this repository deliberately does NOT make

- DSCNS does **not** "solve" catastrophic forgetting.
- DSCNS is **not** an AGI system and does **not** exhibit artificial
  consciousness or human-level intelligence.
- The architecture is *inspired by* episodic/semantic/procedural memory and
  modular cognition; it does **not** model human brain mechanisms.
- No novelty is claimed beyond what is established by the experiments in
  this repository; a full literature review has not been published here.
