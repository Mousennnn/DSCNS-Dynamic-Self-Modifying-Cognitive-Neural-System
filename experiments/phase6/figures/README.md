# Phase 6 Figures

## Figure Index

| Figure ID | Filename | Purpose | Data Source | Generation Script |
|---|---|---|---|---|
| P6-01 | v060_p6_condition_comparison.png | Bar chart comparing key metrics across all 11 experimental conditions | experiments/phase6/raw/ | scripts/plotting/plot_phase6.py |
| P6-02 | v060_p6_training_curve.png | CE loss trajectory across 450 rounds for each condition (mean ± std across 5 seeds) | experiments/phase6/raw/ | scripts/plotting/plot_phase6.py |
| P6-03 | v060_p6_parameter_drift.png | Cumulative parameter drift over 450 rounds | experiments/phase6/raw/ | scripts/plotting/plot_phase6.py |
| P6-04 | v060_p6_safety_risk.png | Safety envelope risk level over rounds | experiments/phase6/raw/ | scripts/plotting/plot_phase6.py |
| P6-05 | v060_p6_exploration_rate.png | Adaptive exploration rate (ε) over rounds | experiments/phase6/raw/ | scripts/plotting/plot_phase6.py |

## Figure Descriptions

### P6-01: Condition Comparison
Five bar charts showing SRR, RFR_similar, EAR, target_accuracy, and magnitude_correlation across all 11 conditions. Error bars show standard deviation across 5 seeds.

**Key observation:** RandomMemory has lowest RFR (0.261), suggesting random memory retrieval paradoxically reduces repeated failures. FrozenPolicy has highest target_accuracy (0.843) because the policy doesn't change. Random has lowest target_accuracy (0.230), confirming the policy is meaningful.

### P6-02: Training Curve
Line plot of CE loss over 450 rounds. All conditions show similar loss trajectories (increasing from ~10 to ~18-19), indicating the base model's performance degrades under continuous self-modification regardless of condition.

### P6-03: Parameter Drift
Shows cumulative parameter drift from initial state. All conditions show similar drift (~3945), indicating the magnitude of parameter change is consistent across conditions.

### P6-04: Safety Risk
Risk level stays at 1.0 throughout, indicating the safety envelope is consistently at maximum risk threshold. This suggests the parameter drift consistently exceeds safe bounds.

### P6-05: Exploration Rate
Shows adaptive exploration rate for conditions with exploration enabled. NoExploration stays at 0.0. Other conditions show ε varying around 0.3.

## Regenerating Figures

```bash
python scripts/plotting/plot_phase6.py \
    --input experiments/phase6 \
    --output experiments/phase6/figures
```

Requires: matplotlib, numpy
