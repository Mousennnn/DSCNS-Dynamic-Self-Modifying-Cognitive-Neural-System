# DSCNS v0.6.0 / Phase 6 Analysis Report

## Acceptance Criteria

- **Minimum Pass**: PASS — D_policy > 0
- **Mechanism Pass**: PASS — Target accuracy > chance AND magnitude correlation > 0
- **Strong Pass**: PASS — D_policy > 0 AND RFR_Full < RFR_NoMemory
- **Full Pass**: FAIL — Strong Pass + EAR > 0

## Evidence Matrix

| Causal Link | Evidence | Status |
|---|---|---|
| Experience -> Policy | D_policy > 0 | SUPPORTED |
| Policy -> Target | target_accuracy=0.439 | SUPPORTED |
| Policy -> Magnitude | magnitude_corr=1.000 | SUPPORTED |
| Modification -> Outcome | RFR_Full=0.410 vs RFR_NoMem=0.445 | SUPPORTED |
| Outcome -> Credit | credit_mean=0.0000 | PARTIAL |
| Credit -> Policy | EAR=0.0000 | NOT ESTABLISHED |
| Full Closed Loop | all links combined | NOT ESTABLISHED |

## Condition Comparison

| Condition | SRR | RFR_similar | EAR | Target_Acc | Mag_Corr | Policy_MI | Net_Drift |
|---|---|---|---|---|---|---|---|
| FrozenPolicy | 0.000 | 0.472 | 0.000 | 0.843 | 1.000 | 0.0466 | 3944.5 |
| FullPolicy | 0.000 | 0.410 | 0.000 | 0.439 | 1.000 | 0.0423 | 3945.5 |
| NoAlternatives | 0.000 | 0.429 | 0.000 | 1.000 | 1.000 | 0.2912 | 3945.9 |
| NoCredit | 0.000 | 0.410 | 0.000 | 0.439 | 1.000 | 0.0423 | 3945.5 |
| NoExploration | 0.000 | 0.385 | 0.000 | 0.433 | 1.000 | 0.0809 | 3947.3 |
| NoMemory | 0.000 | 0.445 | 0.000 | 0.506 | 1.000 | 0.0672 | 3945.7 |
| NoOutcomeReward | 0.000 | 0.410 | 0.000 | 0.439 | 1.000 | 0.0423 | 3945.5 |
| Oracle | 0.000 | 0.410 | 0.000 | 0.437 | 1.000 | 0.0406 | 3945.5 |
| Random | 0.000 | 0.455 | 0.000 | 0.230 | 1.000 | 0.0000 | 3946.9 |
| RandomMemory | 0.000 | 0.261 | 0.000 | 0.454 | 1.000 | 0.0443 | 3947.9 |
| ZeroMemory | 0.000 | 0.445 | 0.000 | 0.506 | 1.000 | 0.0672 | 3945.7 |

## Policy Divergence (FullPolicy vs each condition)

| Pair | KL | JS | Cosine |
|---|---|---|---|
| FullPolicy_vs_FrozenPolicy | 0.0064 | 0.0016 | 0.9937 |
| FullPolicy_vs_NoAlternatives | 0.0001 | 0.0000 | 0.9999 |
| FullPolicy_vs_NoCredit | 0.0000 | 0.0000 | 1.0000 |
| FullPolicy_vs_NoExploration | 0.0001 | 0.0000 | 0.9999 |
| FullPolicy_vs_NoMemory | 0.0003 | 0.0001 | 0.9997 |
| FullPolicy_vs_NoOutcomeReward | 0.0000 | 0.0000 | 1.0000 |
| FullPolicy_vs_Oracle | 0.0000 | 0.0000 | 1.0000 |
| FullPolicy_vs_RandomMemory | 0.0002 | 0.0000 | 0.9998 |
| FullPolicy_vs_ZeroMemory | 0.0003 | 0.0001 | 0.9997 |
