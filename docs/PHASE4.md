# Phase 4 — Learned Model-Driven Structural Self-Adaptation

> Status: **Early research prototype (v0.2.0)** · Design report modification
> proposal: *"DSCNS 自主神经结构自修改机制"*

## 1. Motivation

Phase 3 demonstrated that DSCNS can *modify its network population at
runtime* (split / merge / connect). However, in Phase 3 every structural
decision is produced by the hand-coded rule engine inside `StructureEvolver`
(`should_split()`, `should_merge()`, `update_connections()`). That is:

```
model state → human-defined metrics → human-defined thresholds → StructureEvolver → mutation
```

Phase 4 moves the **decision power** — *whether* to modify, *what* to modify,
*where*, and *how much* — from the rule engine to a small **trainable neural
policy**. The rule engine is kept only as a **hard safety-constraint layer**
and as the Stage-A (imitation) teacher. The target flow is:

```
model state → Self-State Representation → Learned Self-Modification Policy
→ ArchitectureAction → safety constraints → candidate architecture
→ short adaptation → regression evaluation → accept / rollback → reward → policy update
```

No prompts, no natural-language modification plans, no generated code, no
`exec()`: the policy produces structure decisions purely by forward
propagation of learned parameters.

## 2. Scope boundary (important)

DSCNS "networks" are shared frozen GPT-2 base + independent LoRA adapters.
The self-modification object in Phase 4 is therefore the **cognitive-network
/ adapter population** (which networks exist, how they are connected, how
they are created/removed), *not* the internal layer graph of the GPT-2
transformer. Extending learned self-modification to layer-level topology is
future work.

## 3. Architecture

### 3.1 Action space (proposal §3.2)

The policy chooses from a finite action space; each action is carried by the
`ArchitectureAction` dataclass (`dscns/self_modification.py`):

| Operation | Target | Semantics |
|---|---|---|
| `no_op` | — | no structural change |
| `expand` | domain | add a new network for the weakest domain |
| `contract` | network | remove one network |
| `split` | network | N_i → {N_i^a, N_i^b} (k-means task partition) |
| `merge` | network i + j | merge two networks into one |
| `connect` | network i + j | establish a dynamic connection `w = magnitude` |
| `disconnect` | network i + j | remove a connection |

`magnitude ∈ [0,1]` scales the operation (e.g. connection weight);
`confidence` is the sampled probability of the chosen action.

### 3.2 Self-state encoder (proposal §5)

`SelfStateEncoder` (Linear → LayerNorm → GELU → MLP → `z_self`, dim 32)
maps a fixed-size **global self-state vector** (47 dims) computed from the
system's own statistics:

- mean / max / std of per-network competence, uncertainty, task diversity,
  internalized-knowledge count, trust (15)
- population size / budget, connection density, mean connection weight (3)
- learning progress, recent probe performance, mean forgetting (3)
- domain-coverage mean / min (2)
- representation diversity / redundancy (2)
- mean specialization (1)
- elapsed-round fraction, rounds since last change (2)
- parameter utilization, recent modification success rate, recent mean
  reward, pending-modification flag (4)
- per-domain coverage / performance / forgetting (5 + 5 + 5)

Target selection uses a pointer-style scoring of the **per-network feature
matrix** (12 features × n) and the per-domain matrix (5 × 5), so the policy
works for a varying number of networks.

### 3.3 Policy heads (proposal §4)

`SelfModificationPolicy` outputs:

- `P(action)` over the 7 operations (temperature-scaled softmax);
- `P(target | z, F_net)` and `P(secondary_target | z, F_net)` for
  network-targeting operations;
- `P(domain | z, F_dom)` for `expand`;
- `magnitude` (sigmoid head);
- `value` — the learned critic baseline for REINFORCE.

### 3.4 StructureEvolver split (proposal §6)

`StructureEvolver` keeps the *execution* capability (`split_network`,
`merge_networks`, `expand_network`, `contract_network`, connection updates)
and its rule triggers. In Phase 4 the rules are used as:

- **Stage-A teacher** — generate imitation labels for the policy, and
- **hard safety constraints** — `validate_action()` blocks illegal actions
  (unknown targets, budget exceeded, insufficient split data, merging the
  same network, duplicate connections, etc.).

### 3.5 Candidate evaluation & rollback (proposal §10)

Structural changes are treated like internalization trials:

```
snapshot → apply(action) → short adaptation (window W rounds)
→ regression evaluation (system probe performance)
→ accepted (probe not degraded beyond tolerance) or rollback (restore snapshot)
```

The rollback restores network bookkeeping, adapter weights and the
connection table; adapters created by the candidate are orphaned (peft 0.12
has no `delete_adapter`) and simply removed from the network registry.

### 3.6 Modification reward (proposal §11)

```
Reward = Δperf_marginal − λ1·max(0, Δforgetting) − λ2·param_growth
        − λ3·compute_cost − λ4·instability
```

where `Δperf_marginal` is the probe-performance delta over the window minus
the expected per-round learning delta (no-modification counterfactual),
`param_growth` is the relative adapter-parameter increase, `compute_cost` is
a per-operation constant, and `instability` is the std of probe performance
over the window. Default lambdas: 0.5 / 0.3 / 0.1 / 0.3.

### 3.7 Policy learning (proposal §8, §9, §12)

- **Stage A — imitation.** During the warm-up rounds the rule engine decides
  (single-action protocol) and the policy is trained by supervised
  cross-entropy on `(state → rule action)` plus target/magnitude losses.
- **Stage B — result-driven REINFORCE.** The policy proposes actions; after
  each adaptation window a reward is attached; REINFORCE with the learned
  value baseline (plus a small entropy bonus) updates the policy. Stage-B
  exploration uses ε-greedy sampling (ε = 0.15).

### 3.8 Modification memory (proposal §13)

`ModificationMemory` (`dscns/modification_memory.py`) records every
modification attempt: state before, action, acceptance, reward and all
reward components — the experience that closes the loop
experience → meta-cognition → structural adaptation → experience.

## 4. Files

| File | Content |
|---|---|
| `dscns/self_modification.py` | `ArchitectureAction`, `SelfStateEncoder`, `SelfModificationPolicy`, `SelfModificationController` (state collection, rule decision, imitation, REINFORCE, reward, trace) |
| `dscns/modification_memory.py` | `ModificationMemory`, `ModificationRecord` |
| `dscns/evolution.py` | `validate_action()`, `execute_action()`, `expand_network()`, `contract_network()`, serialized split names |
| `dscns/networks.py` | `CognitiveNetwork.get_self_state()` |
| `dscns/system.py` | controller dispatch in `evolve_structure()`, pending-modification window, accept/rollback, reward |
| `scripts/run_phase4.py` | final comparison: rule vs learned vs fixed |

## 5. Experiment (final comparison, proposal §19)

On a shifted stream — general(4) → code(4) → mixed_code(4) → science(4),
16 rounds, 32 experiences/round — three arms are compared:

- **fixed** — 5 networks, no structural evolution (control);
- **rule** — rule-based controller (single-action protocol), same
  candidate → evaluate → accept/rollback machinery;
- **learned** — Stage A imitation (8 rounds) then Stage B policy-driven
  (8 rounds).

Metrics: per-domain performance matrix, AF / FWT / CLS, structural metrics,
action distribution (rule vs policy), acceptance rate, modification reward,
policy entropy, imitation/RL losses.

### 5.1 Results (filled from `experiments/phase4`)

Seed 42, 16 rounds (general(4) → code(4) → mixed_code(4) → science(4)),
32 experiences/round, eval 48/domain. Warm-up = 8 rounds, adaptation
window = 3 rounds.

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

Structural activity:

- **rule** — merge (r3, N1+N2), merge (r6, N4+N5), merge (r9, N1+N4),
  split (r12, N3); all accepted. Aggressive merging hurt math retention
  (math final 0.0361) and raised AF to 0.0099.
- **learned** — Stage A (rule imitation): merge (r3), merge (r6);
  imitation loss 1.99 → 1.55. Stage B (policy): merge (r11, N1+N3,
  accepted, reward −0.009), 4 no-ops, and 2 invalid attempts blocked by the
  safety layer (disconnect without an existing connection; split with low
  diversity). Policy action entropy 1.94 → 1.77; mean reward −0.0031.

Interpretation (single seed, prototype scale): the learned controller
achieved the best final mean / CLS / code adaptation on this run and lower
forgetting than the rule controller; the differences are small and not
statistically established. The learned policy modified more conservatively
than the raw rule engine — a plausible response to slightly negative
modification rewards.

## 6. Acceptance criteria (proposal §19)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Policy parameters obtained by training | ✅ | Stage A imitation (loss 1.99→1.55) + Stage B REINFORCE (see `phase4_learning.png`) |
| 2 | Policy input is the model's own state | ✅ | 47-dim self-state vector built from network/meta statistics |
| 3 | Policy outputs structural modification actions | ✅ | Stage B proposed `merge N1+N3` (r11) and 2 invalid ops (blocked by safety) |
| 4 | Actions are not decided by fixed thresholds | ✅ | action logits/magnitude from policy forward propagation |
| 5 | Modification results produce feedback | ✅ | reward = Δperf − λ·(forgetting/params/compute/instability) |
| 6 | Policy updates from the feedback | ✅ | REINFORCE with value baseline; negative merge rewards shifted behavior |
| 7 | Modification behavior changes over time | ✅ | action entropy 1.94 → 1.77; policy became more conservative than rules |
| 8 | Removing the rule engine's decision logic still yields structural changes | ✅ | Stage B (r8–15) ran with rule decisions off; merge at r11 executed |

## 7. Honest limitations

- Tiny policy, tiny RL budget (≤ 8 Stage-B rounds in the default run):
  the RL signal is a proof-of-concept, not evidence of scalable
  architecture search.
- The reward uses probe-set performance and eval-set performance for state
  features (same convention as Phase 3's rule triggers).
- "Learned" here means learned *when/what/where/how-much to modify the
  network population*; it does **not** mean learned modification of the
  transformer's internal layer graph.
- Phase 4 does not claim to solve architecture search, continual learning,
  or AGI; results are prototype-scale and preliminary.
