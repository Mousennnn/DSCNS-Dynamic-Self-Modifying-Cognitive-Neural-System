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
  (Phase 4 adds a *learned* structural self-modification policy; see §7.)
- Relevance and trust-weight functions are hand-designed heuristics.
- The semantic memory is a lightweight knowledge graph, not a learned
  knowledge base.
- The meta-cognitive layer is a rule-based controller, not a learned one.

## 4. Hypotheses status (as of v0.1.0; Phase 4 in §7 and `docs/PHASE4.md`)

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

## 7. Phase 4 — learned structural self-adaptation (v0.2.0)

Phase 4 (see `docs/PHASE4.md`) moves structural-modification *decisions*
from the rule engine to a small trainable policy. Its limitations:

- **Tiny policy and tiny RL budget.** The policy is a few hundred parameters
  and the default run has at most 8 Stage-B rounds; the REINFORCE signal is a
  proof-of-concept, **not** evidence of scalable architecture search.
- **Action space is closed.** The policy chooses among 7 predefined
  operations; it cannot invent new operations or edit code/graphs directly
  (an explicit design boundary).
- **Self-modification object is the network/adapter population**, not the
  transformer's internal layer graph.
- **Evaluation/state leakage conventions.** State features use eval-set
  performance and the reward uses probe-set performance — the same
  convention as the Phase 3 rule triggers; decisions are not made on the
  final reported metrics alone.
- **Imitation dependence.** The learned controller starts from rule-imitation
  (Stage A); without the imitation prior the RL stage would be much weaker.
- **Rollback is partial in one case:** after a rejected merge, the surviving
  network keeps the learning performed during the adaptation window (the
  rejected partner's bookkeeping is fully restored).
- peft 0.12 has no `delete_adapter`; rolled-back adapters remain orphaned in
  the PeftModel (unused, small memory cost).
- **No claim** that learned self-modification outperforms rule-based or
  fixed topology — the comparison in `experiments/phase4` is the evidence
  base, including negative or mixed outcomes.

## 8. Phase 5 — intrinsic parameter self-modification (v0.3.0)

Phase 5 (see `docs/PHASE5.md`) adds an `IntrinsicPlasticityModule` inside
each cognitive network that maps internal state to a continuous parameter
change (θ → h → Δθ → θ'). Its limitations:

- **Parameter sensing is a 4-dimensional statistic** `[mean, std, min, max]`
  of the adapter weights — an acknowledged low-dimensional approximation,
  not full-parameter conditioning (the report lists this as a deliberate
  P5-B limit).
- **Trigger is external and fixed-frequency** — the network does not decide
  *when* to modify itself (Level 4, explicitly out of P5 scope).
- **Validation and rollback are experiment-controller duties**, not model
  mechanisms — the report is explicit that this must not be described as the
  model "deciding" to undo its own change.
- **Δθ application scope is partial:** the generated low-rank Δθ (768→16→768)
  is applied to hidden-dimension LoRA projections only; c_attn's
  QKV-concatenated up-projection (3H, r) and the MLP input projection
  (r, 3H) are skipped because their shapes do not match the generated delta.
- **Delta generation is off the gradient path** (`no_grad`); P5-C learning is
  offline (reward-weighted imitation of successful deltas) rather than
  end-to-end differentiable self-modification — an explicit scope boundary.
- **P5-C budget is tiny:** ≤100 recorded cases, a handful of training calls,
  and rewards near zero at `plasticity_alpha=0.01`; any improvement in P_φ is
  expected to be small and is **not** claimed.
- **Closed loop was validated at prototype scale only:** GPT-2 small, single
  LoRA rank, single seed, 20 rounds. No claim that the loop generalizes to
  other scales, models, or seeds.
- **Performance is descriptive, not probative:** P5 explicitly does not use
  performance gains as evidence of self-modification; the small arm
  differences (≈0.00x, single seed) are not statistically established.
- **Meta vector is padded** to 32 dims with 13 populated features — the
  remaining capacity is reserved for future meta signals.
- **No claim** of a model that "wants" or "decides" to change itself, and no
  claim that intrinsic modification improves continual learning — the
  experiments in `experiments/phase5` are the (conservative) evidence base.
