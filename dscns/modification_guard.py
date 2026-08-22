"""Modification Safety Envelope (v0.6.0 / Phase 6).

Prevents dangerous parameter drift during long-term self-modification.

Monitors:
    - Policy entropy (collapse detection)
    - Policy KL between rounds (instability detection)
    - Parameter norm (explosion detection)
    - Probe drift (catastrophic forgetting)
    - Outcome stability (degradation detection)

When risk is detected:
    - Reduces modification magnitude (m_t × risk_factor)
    - NEVER sets m_t = 0 (must always modify at least a little)

Components:
  ModificationGuard       -- safety envelope manager
  SafetyState             -- current safety status
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SafetyState:
    """Current safety envelope status."""
    risk_level: float = 0.0          # 0 = safe, 1 = critical
    risk_factors: Dict[str, float] = field(default_factory=dict)
    magnitude_scale: float = 1.0     # applied to m_t (always > 0)
    intervention_count: int = 0
    last_intervention_round: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "magnitude_scale": self.magnitude_scale,
            "intervention_count": self.intervention_count,
            "warnings": self.warnings[-5:],  # last 5
        }


class ModificationGuard:
    """Safety envelope for self-modification.

    Maintains:
        θ_stable: reference stable parameter state
        D_safe: maximum allowed drift from stable state

    When ||θ_t - θ_stable|| > threshold:
        m_t = m_t × risk_factor  (reduced, NEVER zero)

    Also monitors:
        - Policy entropy: if too low → policy collapse
        - Policy KL: if too high → policy instability
        - Parameter norm: if too high → parameter explosion
        - Probe performance: if too low → catastrophic forgetting
    """

    def __init__(
        self,
        max_param_drift: float = 100.0,
        max_param_norm: float = 5000.0,
        min_entropy: float = 0.1,
        max_policy_kl: float = 2.0,
        min_probe_performance: float = 0.01,
        risk_threshold: float = 0.7,
        min_magnitude_scale: float = 0.1,
        enabled: bool = True,
    ):
        self.max_param_drift = max_param_drift
        self.max_param_norm = max_param_norm
        self.min_entropy = min_entropy
        self.max_policy_kl = max_policy_kl
        self.min_probe_performance = min_probe_performance
        self.risk_threshold = risk_threshold
        self.min_magnitude_scale = min_magnitude_scale
        self.enabled = enabled

        # state
        self.theta_stable: Optional[Dict[str, Any]] = None
        self.state = SafetyState()
        self.history: List[Dict[str, Any]] = []
        self._param_norm_baseline: float = 0.0

    def set_stable_state(self, param_state: Dict[str, Any],
                         param_norm: float = 0.0) -> None:
        """Set the reference stable parameter state."""
        self.theta_stable = {k: v.clone() if hasattr(v, 'clone') else v
                             for k, v in param_state.items()}
        self._param_norm_baseline = param_norm

    def check_safety(
        self,
        round_id: int,
        param_norm: float = 0.0,
        param_drift: float = 0.0,
        policy_entropy: float = 0.5,
        policy_kl: float = 0.0,
        probe_performance: float = 0.5,
    ) -> SafetyState:
        """Evaluate safety state and compute magnitude scale.

        Args:
            round_id: current round.
            param_norm: current parameter norm.
            param_drift: drift from stable state.
            policy_entropy: current policy entropy.
            policy_kl: KL divergence from previous policy.
            probe_performance: current probe performance.

        Returns:
            Updated SafetyState with risk info and magnitude scale.
        """
        if not self.enabled:
            self.state = SafetyState(risk_level=0.0, magnitude_scale=1.0)
            return self.state

        risk_factors = {}
        warnings = []

        # 1. Parameter drift
        if self._param_norm_baseline > 0:
            drift_ratio = param_drift / max(self._param_norm_baseline, 1e-6)
            risk_factors["param_drift"] = float(np.clip(drift_ratio, 0, 1))
            if drift_ratio > 0.5:
                warnings.append(f"High param drift: {drift_ratio:.3f}")
        elif param_drift > self.max_param_drift:
            risk_factors["param_drift"] = float(np.clip(
                param_drift / self.max_param_drift, 0, 1))
            warnings.append(f"Param drift {param_drift:.1f} > max {self.max_param_drift}")

        # 2. Parameter norm
        if param_norm > self.max_param_norm:
            risk_factors["param_norm"] = float(np.clip(
                param_norm / self.max_param_norm, 0, 1))
            warnings.append(f"Param norm {param_norm:.1f} > max {self.max_param_norm}")

        # 3. Policy entropy (collapse = bad)
        if policy_entropy < self.min_entropy:
            risk_factors["policy_collapse"] = float(np.clip(
                (self.min_entropy - policy_entropy) / self.min_entropy, 0, 1))
            warnings.append(f"Low policy entropy: {policy_entropy:.3f}")

        # 4. Policy KL (too high = instability)
        if policy_kl > self.max_policy_kl:
            risk_factors["policy_instability"] = float(np.clip(
                policy_kl / self.max_policy_kl, 0, 1))
            warnings.append(f"High policy KL: {policy_kl:.3f}")

        # 5. Probe performance (too low = catastrophic forgetting)
        if probe_performance < self.min_probe_performance:
            risk_factors["probe_drift"] = float(np.clip(
                (self.min_probe_performance - probe_performance) /
                max(self.min_probe_performance, 1e-6), 0, 1))
            warnings.append(f"Low probe perf: {probe_performance:.3f}")

        # compute aggregate risk
        if risk_factors:
            risk_level = float(np.mean(list(risk_factors.values())))
        else:
            risk_level = 0.0

        # compute magnitude scale (always > 0)
        if risk_level > self.risk_threshold:
            # smooth scaling: interpolate between min_scale and 1.0
            t = (risk_level - self.risk_threshold) / (1.0 - self.risk_threshold)
            t = float(np.clip(t, 0, 1))
            mag_scale = self.min_magnitude_scale + (1.0 - self.min_magnitude_scale) * (1.0 - t)
            mag_scale = max(mag_scale, self.min_magnitude_scale)
        else:
            mag_scale = 1.0

        self.state = SafetyState(
            risk_level=risk_level,
            risk_factors=risk_factors,
            magnitude_scale=mag_scale,
            intervention_count=self.state.intervention_count + (1 if mag_scale < 1.0 else 0),
            last_intervention_round=round_id if mag_scale < 1.0 else self.state.last_intervention_round,
            warnings=warnings,
        )

        self.history.append({
            "round_id": round_id,
            "risk_level": risk_level,
            "magnitude_scale": mag_scale,
            "risk_factors": risk_factors,
        })

        return self.state

    def apply_guard(self, magnitude: float) -> float:
        """Apply safety guard to a magnitude value.

        Always returns > 0 (never kills modification completely).
        """
        if not self.enabled:
            return magnitude
        return max(magnitude * self.state.magnitude_scale, 1e-6)

    def is_safe(self) -> bool:
        """Check if the system is in a safe state."""
        return self.state.risk_level < self.risk_threshold

    def risk_trajectory(self) -> List[Dict[str, Any]]:
        """Return risk level over time."""
        return [{"round_id": h["round_id"], "risk_level": h["risk_level"],
                 "magnitude_scale": h["magnitude_scale"]}
                for h in self.history]

    def summary(self) -> Dict[str, Any]:
        """Summary of safety state."""
        if not self.history:
            return {"n_checks": 0}
        risks = [h["risk_level"] for h in self.history]
        scales = [h["magnitude_scale"] for h in self.history]
        return {
            "n_checks": len(self.history),
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "mean_scale": float(np.mean(scales)),
            "min_scale": float(np.min(scales)),
            "n_interventions": self.state.intervention_count,
            "current_risk": self.state.risk_level,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "thresholds": {
                "max_param_drift": self.max_param_drift,
                "max_param_norm": self.max_param_norm,
                "min_entropy": self.min_entropy,
                "max_policy_kl": self.max_policy_kl,
                "min_probe_performance": self.min_probe_performance,
                "risk_threshold": self.risk_threshold,
            },
            "state": self.state.to_dict(),
            "summary": self.summary(),
        }
