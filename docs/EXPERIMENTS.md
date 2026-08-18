# DSCNS Experiments

This document describes the experimental setup, parameters, metrics, and
results of the DSCNS research prototype (Phases 0–3). All raw results are
stored under `experiments/` (JSON + plots) and are part of the research
record — including negative and mixed findings.

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.8.16 |
| PyTorch | 1.13.1+cu117 (CUDA 11.7) |
| transformers / datasets / peft / accelerate | 4.45.2 / 2.21.0 / 0.12.0 / 0.34.2 |
| huggingface_hub / tokenizers | 0.25.2 / 0.20.1 |
| GPU | NVIDIA RTX 3070 Ti Laptop 8 GB |

## 2. Base model (Phase 0)

GPT-2 small (124M parameters, within the design report's 100M–500M target
range). The base model is **frozen**; all learning happens in per-network
LoRA adapters (rank r=16, α=32, dropout 0.1, target modules `c_attn`/`c_proj`).

## 3. Experience stream

24 rounds, 32 experiences per round:

| Rounds | Phase | Domain | Source |
|---|---|---|---|
| 1–4 | general | 常识 | Wikitext-103 (Wikipedia) |
| 5–8 | math | 数学 | GSM8K |
| 9–12 | logic | 逻辑 | MATH-500 (fallback for gated competition_math) |
| 13–16 | code | 代码 | HumanEval |
| 17–20 | science | 科学 | SciQ |
| 21–24 | mixed | 混合 | sample across all domains |

Eval/probe sets (64/16 per domain) are held out and disjoint from training.

## 4. Conditions and fairness

| Mode | Description | Per-round gradient budget |
|---|---|---|
| Control | sequential LoRA fine-tuning, no gating | 8 unconditional steps |
| Exp1 | single network + candidate evaluation + selective progressive internalization | ≤ 8 regression-gated trials |
| Exp2 | 5 networks + cross-network verification + meta-cognitive decisions | ≤ 8 gated trials (≤2 networks/round) |

**Budget parity:** all modes share the same per-round gradient-step ceiling
(`max_grad_steps_per_round = 8`). Exp1/Exp2 additionally pay verification
costs (probe regression tests) that Control does not pay.

### Key parameters

- internalization: `steps=5`, `max_alpha=1.0` (α ramp 0.2→1.0 × lr 5e-4),
  `tolerance=0.02` (probe-performance drop allowed before rollback)
- verification: `conflict_threshold=0.4`, `trust_lr=0.05`, initial trust 0.5
- meta: `acceptance_threshold=0.25`, `store_threshold=0.30`
- evolution (Phase 3): stabilization period 6 rounds, ≤1 split/merge per
  round, merge requires overlap>0.97 ∧ co-activation≥8 ∧ representation
  similarity>0.9
- sequence length 192 tokens, batch 8, AdamW lr 5e-4

## 5. Metrics

- **Performance:** exp(−masked-CE-loss) per domain (masked per-token CE,
  padding tokens excluded), evaluated on held-out eval sets (64/domain).
- **Average Forgetting (AF):** per-domain (peak − final) averaged
  (report §10.3).
- **Forward Transfer (FWT):** per-domain (perf at end of own phase − initial)
  averaged.
- **Continual Learning Score (CLS):** mean(final) − AF.
- **Generation accuracy:** exact-answer match with greedy generation on
  SciQ / GSM8K samples (reported separately; near zero for GPT-2 scale).

## 6. Results

### Phase 1 — multi-network continual learning (Control / Exp1 / Exp2)

| Metric | Control | Exp1 | Exp2 |
|---|---|---|---|
| Average Forgetting (AF) ↓ | 0.0037 | **0.0010** | 0.0013 |
| Forward Transfer (FWT) ↑ | 0.0002 | −0.0041 | **+0.0013** |
| Continual Learning Score (CLS) ↑ | **0.0868** | 0.0735 | 0.0565 |
| Mean acquisition ↑ | 0.0942 | 0.0725 | 0.0548 |
| Mean retention ↑ | **0.0905** | 0.0745 | 0.0578 |

Per-domain forgetting (final vs peak):

| Domain | Control | Exp1 | Exp2 |
|---|---|---|---|
| general | −0.23 | −0.02 | **+0.03** |
| math | −1.82 | −0.45 | **−0.12** |
| logic | −0.57 | −0.28 | **−0.11** |
| code | −1.82 | −1.05 | **−0.48** |
| science | −3.65 | −3.27 | **−1.66** |

Exp2 system statistics: bus traffic `{broadcast: 760, confidence: 3800,
query: 760, update_notify: 48}`; final trust weights
`{N1: 0.90, N4: 0.80, N3: 0.70, N2: 0.65, N5: 0.65}`.

**Interpretation:** progressive internalization (Exp1/Exp2) consistently
reduced forgetting; Exp2 was the only mode with positive forward transfer.
However, under the strict budget parity, neither multi-network configuration
reached Control's absolute performance — multi-network collaboration does not
automatically improve absolute performance (Finding 1).

### Phase 2 — active learning (random / uncertainty / info_gain / meta)

| Strategy | Final performance | Final coverage |
|---|---|---|
| random (baseline) | 0.0572 | 0.256 |
| uncertainty | 0.0466 | 0.256 |
| **info_gain** | **0.0591** | 0.256 |
| meta | 0.0561 | 0.256 |

Information-gain sampling (learner-state uncertainty × novelty) was the best
strategy. Pure uncertainty sampling was worst (it selects the hardest
samples). The meta strategy's weak-domain weighting did not help when all
domains are equally under-covered.

### Phase 3 — structural evolution (fixed vs evolve)

| Metric | fixed | evolve |
|---|---|---|
| Final mean performance (5 domains) | **0.0575** | 0.0533 |
| Code-domain adaptation (round 4→7 Δ) | **+0.0130** | +0.0041 |
| Final network count | 5 | 5 |

Evolution events (all mechanisms fired as designed): merge (round 7, 5→4),
split (round 9, →5), split (round 11, →6), merge + dynamic connections
(round 13, →5). Structural operations introduced short-term performance
perturbation (the design report's Risk-2), and the dynamic system did not
outperform the fixed topology in this experiment.

### Phase 4 — learned structural self-adaptation (fixed vs rule vs learned)

Following the design-report modification proposal *"DSCNS 自主神经结构自修改
机制"*, the structural-modification *decision* is moved from the rule engine
to a small trainable policy (see `docs/PHASE4.md`). Final comparison on a
shifted stream (16 rounds: general(4) → code(4) → mixed_code(4) →
science(4), 32 experiences/round, eval 48/domain):

| Metric | fixed | rule | learned |
|---|---|---|---|
| Final mean performance (5 domains) | 0.0554 | 0.0570 | **0.0585** |
| Average Forgetting (AF) ↓ | **0.0000** | 0.0099 | 0.0029 |
| Forward Transfer (FWT) ↑ | 0.0005 | **0.0009** | 0.0007 |
| Continual Learning Score (CLS) ↑ | 0.0553 | 0.0471 | **0.0556** |
| Code adaptation (round 4→8 Δ) | +0.0224 | +0.0326 | **+0.0336** |
| Final network count | 5 | 2 | 2 |
| Modification success rate | — | 1.00 | 1.00 |
| Mean modification reward | — | — | −0.0031 |

Controller activity:

- **rule**: 4 structural actions (merge r3 N1+N2, merge r6 N4+N5, merge r9
  N1+N4, split r12 N3), all accepted; aggressive merging hurt math retention
  (math final 0.0361) and raised AF to 0.0099.
- **learned**: warm-up (Stage A, r0–7) = rule decisions (merge r3, merge r6)
  with imitation loss 1.99 → 1.55; Stage B (r8–15) = policy decisions:
  merge N1+N3 at r11 (accepted, reward −0.009), 4 no-ops, plus invalid
  disconnect/low-diversity attempts blocked by the safety layer. Policy
  action entropy decreased 1.94 → 1.77 (behavior change over learning);
  mean reward −0.0031 (structural merges yielded small negative marginal
  reward vs the no-modification learning baseline).

**Interpretation (single seed, prototype scale):** on this run the learned
controller achieved the highest final mean, CLS and code adaptation, and
lower forgetting than the rule controller. Its behavior — fewer, more
conservative modifications informed by slightly negative rewards — is a
plausible learned response, but the differences (≈0.003) are **not
statistically established**. The experiment demonstrates the learned
decision loop (state → policy → action → candidate → evaluate → reward →
policy update) and that the policy can produce structural actions with the
rule engine's decision logic switched off.

### Phase 5 — intrinsic parameter self-modification (fixed vs p5b vs p5c + controls)

Following the design report *"DSCNS Phase 5 内生式参数自修改机制"*, each
cognitive network hosts an `IntrinsicPlasticityModule` (a member of the
network, not an external observer) that maps its own internal state to a
continuous parameter change:

```
theta_t -> h_t -> delta_theta_t = P_phi(h_t, stats(theta_t), s_t)
        -> theta_{t+1} = theta_t + alpha * delta_theta_t -> h_{t+1} -> ...
```

Primary evidence is the closed loop itself; performance is recorded as a
descriptive outcome (the report explicitly does **not** require performance
gains for P5). Stream: 20 rounds = general(5) → code(5) → mixed_code(5) →
science(5), 32 experiences/round, sampling with replacement (small HumanEval
pool), external fixed trigger every 4 grad steps (1 plasticity step per
network per round), `plasticity_alpha=0.01`, learnable modulation strength
init 0.05, seed 42.

| Metric | fixed | p5b (intrinsic) | random | constant | shuffled |
|---|---|---|---|---|---|
| Final mean performance (5 domains) | **0.0413** | 0.0412 | 0.0405 | 0.0409 | 0.0407 |
| Average Forgetting (AF) ↓ | **0.0090** | 0.0090 | 0.0097 | 0.0094 | 0.0095 |
| Continual Learning Score (CLS) ↑ | **0.0323** | 0.0322 | 0.0308 | 0.0315 | 0.0312 |
| Plasticity triggers | — | 100 | 100 | 100 | 100 |
| Acceptance rate | — | 1.00 | 1.00 | 1.00 | 1.00 |
| Mean Δθ norm | — | **1.285** | 1.310 | 1.325 | 1.288 |
| Δθ norm variance (across events) | — | **4.8e-3** | 1.3e-2 | 3.3e-3* | 1.8e-3 |
| Prediction-change rate | — | **0.014%** | 0.019% | 0.012% | 0.007% |
| Logits change | — | **0.0074** | 0.0078 | 0.0078 | 0.0081 |

\* the constant arm caches one constant delta *per network*, so its
across-event variance reflects between-network differences, not input
dependence.

**P5-C (adaptive plasticity learning, compute non-parity — reported
separately):** 100 triggers, 100 accepted, 19/20 success cases and 11 offline
training calls per network, mean reward **+9.6e-4** (after the short
adaptation, performance on the trigger texts generally improved). Unweighted
replay MSE 1.35e-5 → 1.90e-5 (real but tiny); the reward-weighted loss is
≈1e-9, so P_φ updates are negligible. p5c's higher final metrics (final mean
0.0552, AF 0.0007, CLS 0.0545) are **confounded by the 60 extra task-learning
steps per network** spent in the adaptation phase and must **not** be
attributed to plasticity learning (see `docs/PHASE5.md` §5.4).

Core-validation evidence (see `docs/PHASE5.md` §4): Tests 1–6 and the P5-A
modulation test all pass — Δθ is non-zero (‖ΔW_A‖≈0.67, ‖ΔW_B‖≈0.69), state
dependent (deterministic under identical input, cross-input difference
≈0.23, random-hidden ablation ≈1.25), transitions parameters (‖θ'−θ‖≈32.7 at
α=1.0), changes behavior (logits diff ≈0.21, prediction change 0.91%),
forms a non-constant non-diverging loop over 5 iterations, and stays stable
over 20 consecutive steps (no NaN, bounded norms, entropy ≈3.87). Negative
controls are distinguishable: intrinsic vs same-norm random deltas differ in
behavioral effect; intrinsic cross-input Δθ variance ≫ 0 vs 0 for a constant
delta; correct state↔delta pairing produces different effects than shuffled
pairing.

**Interpretation (single seed, prototype scale):** P5's core claim — a
repeatable, stable, measurable parameter–state feedback loop exists and is
state-dependent — is supported by the validation suite. Performance
differences between arms (≈0.00x) are not statistically established and are
not the point of P5.

## 7. Known deviations from the design report

1. **MATH dataset:** `hendrycks/competition_math` is gated; the loader falls
   back to the public `HuggingFaceH4/MATH-500` (500 problems).
2. **HumanEval scale:** only 164 problems exist; the code-domain training
   pool is small (84 items) and samples are reused across rounds.
3. **Evaluation metric:** performance = exp(−masked per-token CE). An early
   version also counted padding positions in the CE loss, which compressed
   all values; this was fixed and every experiment was re-run with the final
   metric.
4. **Internalization granularity:** the design report describes per-item
   internalization; for computational feasibility the prototype internalizes
   the accepted candidate group per network per round, keeping the same
   trial → regression → accept/rollback semantics.
5. **Relevance signal:** frozen-base cosine similarity alone is weak in dense
   text-embedding space; the prototype adds a domain-label match bonus
   (the stream's domain labels are observable system input).
6. **Evolution cadence:** a 6-round stabilization period, at most one
   split/merge per round, and conservative thresholds were added (explicit
   implementation of the report's Risk-2 mitigation).
7. **Phase 4 protocol changes:** the Phase 4 rule arm uses a *single
   ArchitectureAction per round* (priority split > merge > connect) so it is
   directly comparable to the learned controller, and both dynamic arms go
   through the same candidate → evaluate → accept/rollback machinery
   (Phase 3 published numbers used direct execution and are unchanged).
8. **Phase 4 reward uses probe sets; state features use eval sets** — the
   same convention as the Phase 3 rule triggers (decisions are not made on
   the final reported metrics alone).
9. **Phase 5 validation uses a relative safety criterion** — accept when
   `loss_after < loss_before + 0.5 nats` (with an absolute perplexity cap as
   a secondary guard) instead of the report's absolute `perplexity < 100`,
   because GPT-2 raw perplexity varies across domains and an absolute cap
   alone rejects valid modifications.
10. **Phase 5 Δθ application scope** — the generated low-rank Δθ
    (768→16→768) is applied to the hidden-dimension LoRA projections
    (c_attn / attention c_proj down, c_proj / mlp c_proj up); c_attn's
    QKV-concatenated output projection (3H, r) and the MLP input projection
    (r, 3H) are skipped (dimension mismatch; documented in LIMITATIONS).
11. **Phase 5 trigger is external and fixed-frequency** (every 4 grad
    steps) — per the report, the trigger is experiment scheduling, not part
    of the intrinsic mechanism.
12. **Phase 5 stream uses sampling with replacement** for all domains (the
    report's 20-round length exceeds the small HumanEval pool without
    reuse; P5 focuses on the closed loop, not stream curation).

## 8. Reproducibility

- Data/model caches are excluded from git; `scripts/download_model.py` and
  `scripts/common.py` re-fetch them.
- `scripts/resolve_deps.py` + `scripts/download_wheel.py` reproduce the
  dependency bootstrap (resumable downloads, offline install) for restricted
  networks.
- Deterministic seeds (42) for the experience stream; eval/probe splits are
  fixed.
- Results JSONs contain the full per-round log for every mode.
