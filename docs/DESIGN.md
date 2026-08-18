# DSCNS Design

This document summarizes the design of the **Dynamic Self-Modifying Cognitive
Network System (DSCNS)** as specified in the design report v1.0 and as
implemented in this repository. The full Chinese design report is archived at
`docs/DSCNS_design_report.md` (CC BY 4.0).

## 1. Conceptual Model

DSCNS replaces the single "data → parameter update" flow with a closed-loop
cognitive process:

```
Traditional:  data → model → gradient descent → fixed parameters
DSCNS:        experience → multi-network observation → independent evaluation
              → cross-network verification → selective internalization
              → structural reorganization → continual evolution
```

### Ten design principles

| # | Principle | Implementation |
|---|---|---|
| P1 | Receiving information ≠ learning | Experience buffer fully decoupled from parameter updates |
| P2 | Learning ≠ immediate parameter change | Verification precedes any parameter update |
| P3 | Parameter updates must be verified | Regression tests + update-budget constraints |
| P4 | One experience can be observed by many networks | Shared experience broadcast |
| P5 | Observing ≠ internalizing | Knowledge-state levels 0–3 |
| P6 | Knowledge can be shared or locally internalized | Shared memory + local internalization |
| P7 | Forgetting is local, gradual, reactivatable | Importance decay + reactivation |
| P8 | Networks communicate, correct, connect | NetworkCommunicationBus |
| P9 | Network structure is a learning outcome | Split / Merge / Connect / Remove |
| P10 | The system continuously changes itself | S_{t+1} = F(S_t, E_t) |

## 2. System State

The system state at time *t* is defined as

```
S_t = (G_t, Θ_t, M_t, C_t)
```

- `G_t = (V_t, E_t)` — cognitive network graph (nodes = networks, edges = connections)
- `Θ_t = {θ_1, …, θ_k}` — per-network parameters (implemented as per-network
  LoRA adapters on a shared frozen base model)
- `M_t = (M_ep, M_sem, M_proc)` — episodic / semantic / procedural memory
- `C_t` — meta-cognitive state vector

## 3. Core Mechanisms

### 3.1 Knowledge state levels and internalization degree

```
Level 0: external experience (untouched)
Level 1: existence-level cognition (the system knows the knowledge exists)
Level 2: callable knowledge (queryable from other networks / semantic memory)
Level 3: internally internalized (affects network parameters)
```

Each network tracks the internalization degree `I_ij ∈ [0,1]` per knowledge
item *j*.

### 3.2 Independent evaluation

Each network evaluates a candidate knowledge item `K` along four dimensions:

```
Q_i(K) = (R, N, C, I)
```

- `R` — Relevance: cosine similarity between the item's embedding and the
  network's domain embedding (plus a domain-label match bonus in the
  prototype).
- `N` — Novelty: 1 − max similarity to the network's already-internalized
  items.
- `C` — Confidence: base-model evidence (calibrated loss) × source reliability.
- `I` — Importance: relevance × uncertainty-driven utility.

### 3.3 Cross-network verification

Trust-weighted aggregation:

```
C_final(K) = Σ w_i·C_i / Σ w_i,   w_i = trust_i × R_i
```

Conflicts (`max(C_i) − min(C_i) > threshold`) trigger evidence-based
resolution: episodic/semantic evidence retrieval; if evidence is
insufficient, the decision is **deferred** rather than forced. Trust weights
are updated from observed correctness (`±0.05` per event).

### 3.4 Progressive internalization

```
1. Tentative update:   θ_trial = θ + α·Δθ,   α small
2. Regression test:    evaluate on cross-domain probes
3. Accept / rollback:  accept if perf ≥ baseline − tolerance, else roll back
4. Consolidation:      repeat with growing α; stop on detected degradation
```

Update-budget constraint: `‖Δθ‖₂ ≤ ε·‖θ‖₂`.

### 3.5 Active experience selection

```
Score(x) = α·IG(x) + β·U(x) + γ·R(x) − λ·C(x)
```

with information gain estimated from learner-state uncertainty (the
prototype uses the learner's current loss-based uncertainty, so the score
adapts to learning progress).

## 4. Structural Evolution

- **Specialization score:** `S_i^k = (Perf_i^k / AvgPerf^k) × (ActFreq_i^k / TotalAct_i)`
- **Split trigger:** task diversity + representation clustering (k-means) +
  negative-transfer signal above thresholds; children inherit the parent's
  adapter weights and partition its knowledge.
- **Merge trigger:** functional overlap + co-activation frequency +
  representation similarity above thresholds.
- **Dynamic connections:** `w_ij(t) = α·CoActivation_ij(t) + β·InfoFlow_ij(t)`;
  a connection is created when `w_ij > τ`.

Prototype safeguards (per the design report's risk analysis): a stabilization
period before any evolution, at most one split/merge per round, and
conservative thresholds.

## 5. Memory

- **Episodic memory:** time-ordered raw experiences with embedding-based
  recall.
- **Semantic memory:** a lightweight knowledge graph (concept nodes, relation
  edges, confidence, per-network internalization levels).
- **Procedural memory:** successful action sequences per task type.

## 6. Communication

`NetworkMessage(sender, receiver, msg_type, content, timestamp)` with message
types QUERY / RESPONSE / BROADCAST / CONFLICT / CONFIDENCE / UPDATE_NOTIFY /
CORRECTION / MERGE_REQUEST / SPLIT_NOTIFY / META_REPORT, carried by an async
message bus with a fully traceable log and a co-activation matrix.

## 7. Research Hypotheses (as stated in the design report)

- **H1** — Multi-network verified continual learning achieves better
  combined acquisition-and-retention than EWC / replay / fine-tuning.
- **H2** — Dynamic structure outperforms fixed structure under distribution
  shift.
- **H3** — Decoupling experience from parameters plus verification reduces
  error propagation and catastrophic forgetting.
- **H4** — Sharing experience (not parameters) yields complementary local
  representations with less redundancy.

Status of each hypothesis under the current prototype is reported in
`docs/EXPERIMENTS.md` and `docs/LIMITATIONS.md`.
