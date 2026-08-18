# P5 Long-Horizon Experiment v2

**Design report:** "DSCNS Phase 5 实验设计报告 v2" (Random-Control /
No-Modification / Fixed Probe Set / True Net Parameter Drift /
Long-Horizon Operational Loop).

**Frozen P5 implementation:** commit `3208463` (v0.3.0). No P5 core file is
modified by this experiment; `scripts/run_p5_long_horizon.py` only
orchestrates the existing model-side loop
`theta -> h -> delta_theta -> theta'` and logs metrics.

## Two lines

| Line | Groups | Rounds | Question |
|---|---|---|---|
| A. Validation | `p5_150`, `random_control_150`, `no_modification_150`, `baseline` | 150 | Is P5's change real, net, and distinguishable from random modification? |
| B. Extreme run | `p5_3000` | 3000 | What does the closed loop do over a very long horizon (stable / drift / oscillation / divergence / collapse / attractor)? |

## Directory

```
experiments/p5_long_horizon/
├── README.md            # this file
├── config/              # frozen per-group configs (p5_150, random_control_150,
│                        #   no_modification_150, p5_3000)
├── probe_set/           # FROZEN probe set shared by every group
│   ├── probes.json
│   └── probe_0001.txt … probe_0032.txt
├── results/             # per-group metrics.json / metrics.csv / group.json /
│                        #   layer_drift.json + summaries.json
├── checkpoints/         # per-group .pt checkpoints (gitignored):
│                        #   model_state / optimizer_state (null) / p5_state /
│                        #   step / seed / probe_output / parameter_hash
├── plots/               # the 8 required figures
└── report.md            # result-driven report
```

## Key metrics

- **Gross movement** `D_gross(T) = Σ_t ||θ_t − θ_{t−1}||` — how much the
  parameters moved in total.
- **True net drift** `D_net(T) = ||θ_T − θ_0||` — where they ended up.
- **Drift ratio** `R = D_net / (D_gross + ε)` — direction coherence
  (≈1 coherent, ≈0 cancelling).
- **Layer-wise drift** per transformer layer (drift + relative drift) —
  where the modification locus is.
- **Parameter hash** SHA-256 over the adapter weights — proves actual tensor
  change; constant for `no_modification_150` by construction.
- **Probe output drift** on the frozen Probe Set vs t=0 and vs the previous
  evaluation — whether parameter change reaches behavior.
- **Checkpoint state separation** — `model_state` / `optimizer_state`
  (null: this loop has no optimizer or training state) / `p5_state`.

## Fairness of Random-Control

`random_control_150` applies modifications at the same rounds, with the same
per-round magnitude profile (taken from `p5_150`) and through the same
`apply_intrinsic_modification` path; only the *direction* is random. This
isolates "modification at all" from "the direction P5 chooses".

## Execution order (design report §22)

1. Freeze commit + seed (42)
2. Generate & freeze Probe Set (seed 1234, 32 texts, 5 domains)
3. `p5_150` → 4. `random_control_150` → 5. `no_modification_150` →
   6. `baseline` → 7. verify logging → 8. `p5_3000`

Reproduce: `python scripts/run_p5_long_horizon.py` (add `--smoke` for a
6-round-per-group validation run), then
`python scripts/analyze_p5_long_horizon.py`.
