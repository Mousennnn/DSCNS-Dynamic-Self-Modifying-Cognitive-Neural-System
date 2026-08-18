# P5 150-Round Long-Horizon Intrinsic Self-Modification Experiment

> **Result-driven report.** All numbers below come from the actual run
> (`experiments/phase5_long_run/metrics.json` / `metrics.csv` /
> `summary.json`). No data points were removed or modified; no tuning was
> performed; the frozen P5 implementation (commit `3208463`) was used
> unchanged.

---

## 1. Objective

The existing Phase 5 experiments established that the intrinsic
self-modification closed loop

```
theta_t -> h_t -> delta_theta_t = P_phi(h_t, stats(theta_t), s_t)
        -> theta_{t+1} = theta_t + alpha * delta_theta_t -> h_{t+1} -> ...
```

exists, is state-dependent, and stays numerically stable over short
horizons (≤ 20 rounds, see `docs/PHASE5.md`).

This experiment asks the long-horizon question:

> Does the same closed loop keep running, stay numerically stable, and
> remain state-dependent over 150 continuous rounds — and what are the
> long-term dynamics of modification magnitude, parameter drift and
> behavioral change?

This is a **stress test of the frozen P5 mechanism**, not a new
implementation. No P5 core file was modified; the loop was orchestrated by a
new self-contained script (`scripts/run_phase5_long_run.py`) that only calls
the existing model-side methods (`generate_delta`,
`apply_intrinsic_modification`, `snapshot_parameters`,
`_current_params_tensors`, `_logits_for_texts`, `losses_for_texts`).

## 2. Experimental Hypothesis

- **H1**: P5 can produce non-zero parameter modifications in all 150
  continuous rounds (Δθ_t ≠ 0 holds).
- **H2**: Parameter modifications do not cause numerical divergence (no
  NaN / Inf / unbounded norms).
- **H3**: The magnitude of Δθ over the long run may decay, stabilize,
  oscillate or grow — but its change is observable and quantifiable.
- **H4**: Parameter changes keep producing observable behavioral changes.

These are hypotheses to be tested by the data, not pre-written conclusions.

## 3. Experimental Setup

Frozen configuration (also saved in
`experiments/phase5_long_run/config.json`):

| Item | Value |
|---|---|
| Git commit (frozen P5) | `3208463dbb1b9c26f610c69938492990eb0d3af9` (v0.3.0) |
| Python | 3.8.16 |
| PyTorch / CUDA | 1.13.1+cu117 / 11.7 |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop (8 GB) |
| Random seed | 42 (the existing P5 seed) |
| Rounds | 150 (continuous, no re-initialization) |
| Network | N1 (single theta-chain) |
| Model | frozen GPT-2 small (124M), hidden 768, LoRA r=16, α=32, dropout 0.1 |
| Plasticity | alpha = 0.01, rank = 8, meta_dim = 32, modulation strength init 0.05 |
| Data | 8 texts/round, max_len 192, sampling with replacement, 20-round phase pattern (general→code→mixed→science) repeated 7.5× |
| Apply policy | every round, unconditionally (validation/rollback intentionally disabled — see §11) |

Loop mechanics per round: measure `theta_t` behavior (logits/loss/entropy) →
generate `delta_theta_t` via the frozen `generate_delta` → apply
`theta_{t+1} = theta_t + alpha * delta_theta_t` via the frozen
`apply_intrinsic_modification` → measure `theta_{t+1}` behavior → record.

## 4. Metrics

| Metric | Definition |
|---|---|
| `delta_norm` | ‖Δθ_t‖₂ = ‖ΔW_A‖₂ + ‖ΔW_B‖₂ (generated consensus delta) |
| `theta_norm` | ‖θ_t‖₂ over the network's LoRA adapter parameters |
| `relative_delta` | ‖Δθ_t‖₂ / ‖θ_t‖₂ |
| `applied_change` | ‖θ_{t+1} − θ_t‖₂ actually applied to the weights |
| `cumulative_drift` | ‖θ_t − θ_0‖₂ (vs. round-0 snapshot) |
| `logits_diff` | mean \|logits_after − logits_before\| (same inputs) |
| `pred_change` | fraction of token positions whose argmax prediction changed |
| `entropy` | mean softmax entropy of the model output (after modification) |
| `loss` | mean masked per-token CE loss (after modification) |
| `nan_count` / `inf_count` | number of NaN / Inf entries in Δθ and θ |
| `parameter_changed` | whether ‖θ_{t+1} − θ_t‖ > 0 |
| `hidden_mean` / `hidden_std` | statistics of the pooled hidden state used by P_φ |

## 5. Results

All 150 rounds completed. **Zero NaN, zero Inf** (first-anomaly round:
none).

| Quantity | Value |
|---|---|
| Final parameter drift ‖θ₁₅₀ − θ₀‖₂ | **40.995** |
| Mean ‖Δθ‖₂ | **1.2139** |
| Std ‖Δθ‖₂ | 0.0145 |
| Min / Max ‖Δθ‖₂ | 1.1981 / 1.2431 |
| Mean relative modification | 0.0757 |
| Final relative modification | 0.0641 |
| θ norm: first → last | 13.896 → 18.706 (+34.6%) |
| Mean applied change / round | 0.2913 |
| Mean logits difference | 0.00368 |
| Prediction-change rate (mean) | 0.000087 (0.0087%); 11/150 rounds had any flip |
| Entropy: first → last | 4.898 → 2.890 (range 2.606–5.408; never collapsed) |
| Loss: first → last | 3.962 → 2.693 (range 2.492–5.044) |
| NaN / Inf | **0 / 0** |

### Early / middle / late (rounds 1–30 / 31–100 / 101–150)

| Phase | mean ‖Δθ‖ | std ‖Δθ‖ | mean rel. | drift start→end | mean logits | mean entropy | mean loss |
|---|---|---|---|---|---|---|---|
| early (1–30) | 1.2348 | 0.0092 | 0.0864 | 0.29 → 8.14 | 0.00389 | 3.950 | 3.563 |
| middle (31–100) | 1.2145 | 0.0097 | 0.0770 | 8.40 → 27.08 | 0.00341 | 4.042 | 3.795 |
| late (101–150) | 1.2004 | 0.0014 | 0.0674 | 27.36 → 40.99 | 0.00393 | 4.107 | 3.711 |

### Five-window statistics (rounds 1–30 / 31–60 / 61–90 / 91–120 / 121–150)

| Window | ‖Δθ‖ mean / std / min / max | rel. mean | ‖θ‖ mean | logits mean | entropy mean |
|---|---|---|---|---|---|
| 1–30 | 1.2348 / 0.0092 / 1.2169 / 1.2431 | 0.0864 | 14.5 | 0.00389 | 3.950 |
| 31–60 | 1.2241 / 0.0069 / 1.2088 / 1.2362 | 0.0795 | 15.6 | 0.00314 | 4.042 |
| 61–90 | 1.2086 / 0.0062 / 1.1991 / 1.2217 | 0.0735 | 16.6 | 0.00365 | 4.114 |
| 91–120 | 1.2020 / 0.0036 / 1.1981 / 1.2114 | 0.0696 | 17.4 | 0.00336 | 4.069 |
| 121–150 | 1.1997 / 0.0014 / 1.1981 / 1.2028 | 0.0669 | 18.2 | 0.00382 | 4.034 |

### Linear trends (per round, least-squares slope)

| Series | Slope/round |
|---|---|
| ‖Δθ‖ | −2.98e-4 (slow decay) |
| ‖θ‖ | +3.23e-2 (steady growth) |
| logits difference | +2.94e-6 (≈ flat) |
| entropy | +5.47e-4 (≈ flat) |

## 6. Long-Horizon Dynamics

- **Modification magnitude (‖Δθ‖):** mildly **decaying**, not vanishing.
  First-10 mean 1.2239 → last-10 mean 1.1990 (−2.0% over 150 rounds); the
  trend slope is −2.98e-4/round. The decay is smooth and monotonic
  (window means 1.235 → 1.224 → 1.209 → 1.202 → 1.200), with per-window
  standard deviation shrinking (0.0092 → 0.0014), i.e. the loop settles
  into a narrow, stable magnitude band rather than oscillating.
- **No growth, no explosion, no oscillation** of ‖Δθ‖: min/max span is only
  1.198–1.243 (±1.8% around the mean).
- **Parameter norm (‖θ‖):** steady **growth** (+3.23e-2/round; 13.90 → 18.71),
  a slow accumulation consistent with coherent directional modifications.
- **Cumulative drift:** grows **linearly** — per-30-round drift gains are
  7.84, 7.74, 7.91, 8.02, 8.09 (≈ constant ≈ 0.27/round). No acceleration,
  no saturation, no phase transition between early/middle/late windows.
- **Relative modification:** monotonically **decreasing** (0.086 → 0.064)
  because ‖θ‖ grows while ‖Δθ‖ slowly shrinks — this is the expected
  normalization effect, not an instability.
- **Behavioral drift:** logits difference stays ≈ 0.003–0.004 throughout
  (slope ≈ 0); entropy fluctuates round-to-round (2.6–5.4, data-driven,
  different domain batches) but shows no trend and never collapses.
- **Phase transitions:** none observed; the three-phase comparison shows only
  gradual monotonic shifts (Δθ −2.8%, relative −22%, drift linear).

## 7. Stability Analysis

1. **NaN:** 0 across all 150 rounds (0 in Δθ, 0 in θ).
2. **Inf:** 0 across all 150 rounds.
3. **Parameter explosion:** no — ‖θ‖ grew from 13.90 to 18.71 (+34.6%) at a
   constant linear rate; nothing divergent.
4. **Δθ explosion:** no — ‖Δθ‖ stayed in [1.198, 1.243].
5. **Relative modification abnormal growth:** no — it *decreased* from 0.088
   to 0.064.

Conclusion: the frozen P5 loop is numerically stable over 150 rounds.

## 8. Behavioral Analysis

- Parameter modification → behavioral change: every round changes the
  weights (applied_change ≈ 0.29/round) and every round moves the logits
  (mean |Δlogits| ≈ 0.0037, stable across the whole run). The modification
  therefore continues to affect model output at every step — H4 supported.
- The relationship is **proportional in scale**: per-round behavioral change
  is roughly constant while per-round modification magnitude is roughly
  constant, i.e. the behavioral effect tracks the modification scale.
- Caution: this documents a *relationship* between modification magnitude
  and output change. It is **not** evidence of a causal "improvement"
  mechanism — no performance claim is made (P5's stated position).

## 9. Comparison With Existing P5 Short-Horizon Experiment

Reference: `experiments/phase5` (P5-B, 20 rounds, 5 networks,
validation/rollback enabled, alpha = 0.01).

| Quantity | Short run (P5-B, 100 events) | Long run (150 rounds) |
|---|---|---|
| Mean ‖Δθ‖ | 1.2848 ± 0.0693 | 1.2139 ± 0.0145 |
| Relative modification | ~0.09 (init) | 0.086 → 0.064 |
| θ norm growth | 13.98 → 15.81 (+13%) | 13.90 → 18.71 (+34.6%) |
| Prediction-change rate | 0.014% | 0.0087% |
| Logits difference | 0.0074 | 0.0037 |
| NaN / Inf | 0 / 0 | 0 / 0 |
| State dependence | Test 2: input diff 0.231 | loop continues to run on varied inputs; ‖Δθ‖ stays bounded and state-driven (batch composition changes every round) |

The long run is consistent with the short run: same magnitude scale, same
stability, slightly *lower* per-event behavioral perturbation (α identical;
the long run measures the raw loop without adaptation/training between
events, so the weights are closer to the delta-generation state). No short
experiment result was modified or deleted.

## 10. Findings

**Observed (directly measured):**
- 150/150 rounds produced non-zero Δθ (min ‖Δθ‖ = 1.1981 > 0) — H1 holds.
- 0 NaN and 0 Inf over 150 rounds; bounded norms throughout — H2 holds.
- ‖Δθ‖ declined slowly and smoothly (1.235 → 1.200, −2.0% over the run),
  well above zero — H3's "decay" branch, observable and quantifiable.
- ‖θ‖ grew linearly (+34.6%); cumulative drift accumulated linearly
  (~0.27/round) and reached 40.99.
- Every round changed logits (mean 0.0037); 11/150 rounds flipped at least
  one argmax prediction — H4 holds at the "continuous small behavioral
  change" level.
- Drift/Σ‖applied change‖ = 0.938 — the modifications are **~94% coherent**
  (directional accumulation), far above a neutral random walk (~0.07).

**Supported (data-backed statements):**
- The frozen P5 closed loop remains continuous, state-dependent in input,
  numerically stable, and non-vanishing over 150 rounds.
- Long-run dynamics are monotonic (slow Δθ decay, linear θ/drift growth) with
  no oscillation, explosion, or phase transition.

**Not established (this experiment cannot show):**
- That the modification is beneficial (no performance claim; loss varied
  round-to-round with the input domain, 2.5–5.0, with no monotonic trend).
- That the system "learns" or "adapts to improve" — P_φ was fixed (frozen
  module, modulation strength constant at 0.05; no P5-C training here).
- That 150 rounds generalize to other seeds, models, or longer horizons.
- That the system understands why it modifies itself (Level 4, out of scope).

## 11. Limitations

- **Single seed** (42), single network (N1 chain), single model (GPT-2 small),
  single run: no statistical multi-seed validation.
- **Only 150 rounds** — long for this prototype but not a general
  long-horizon claim.
- **Validation/rollback disabled** in this stress test: the raw intrinsic
  mechanism is applied unconditionally every round to expose worst-case
  dynamics; the controller-side safety protocol (P5 design report §8) was
  intentionally not part of this run. This is the strongest stability probe,
  not the P5-B operational loop.
- **Input sampling with replacement**; the 20-round domain pattern repeats.
- **No performance improvement claim**; loss is recorded but is
  domain-driven and shows no trend.
- **No claim of autonomous goal formation, no AGI claim, and no claim that
  long-term stable running equals "true autonomous evolution."** Stable
  closed-loop dynamics are exactly that — a stable, state-dependent
  parameter–state loop — and nothing more.

## 12. Conclusion

**Yes.** The frozen P5 intrinsic parameter self-modification mechanism
(`theta -> h -> delta_theta -> theta'`) ran for **150 continuous rounds**
without re-initialization, produced a non-zero state-dependent Δθ in every
round (mean ‖Δθ‖ = 1.2139, std 0.0145), accumulated measurable parameter
drift (‖θ₁₅₀ − θ₀‖ = 40.99, linear at ~0.27/round), and remained fully
numerically stable (0 NaN, 0 Inf, bounded norms, entropy never collapsed).
The long-horizon dynamics are monotonic: Δθ magnitude slowly decays (−2.0%)
without vanishing, ‖θ‖ grows steadily, and behavioral change persists at a
small constant scale. No oscillation, explosion, or phase transition was
observed. The 150-round stress test therefore confirms the short-horizon P5
conclusions at a longer time scale, within the limits stated in §11.
