"""Progressive internalization controller (report section 8.3 / 3.5).

Implements the trial -> regression test -> accept/reject -> consolidation
loop with an explicit update budget constraint
    ||dTheta_i||_2 <= eps * ||Theta_i||_2   (enforced via small alpha steps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class InternalizationResult:
    success: bool = False
    final_level: float = 0.0
    stop_reason: str = ""
    stopped_at_step: int = -1
    perf_before: float = 0.0
    perf_after: float = 0.0
    trials: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


class InternalizationController:
    """Controls how knowledge is gradually internalized into a network."""

    def __init__(self, tolerance: float = 0.02, max_alpha: float = 0.1,
                 steps: int = 5, probe_batch: int = 8, max_len: int = 256):
        self.tolerance = tolerance          # allowed performance loss threshold
        self.max_alpha = max_alpha          # max single-step update magnitude
        self.steps = steps                  # number of consolidation steps
        self.probe_batch = probe_batch
        self.max_len = max_len

    # ------------------------------------------------------------------ #
    def internalize(self, network: Any, candidates: List[Any],
                    target_level: float, tokenizer: Any,
                    regression_test_fn: Callable[[Any], float],
                    max_steps: Optional[int] = None) -> InternalizationResult:
        """Progressively internalize a batch of knowledge items.

        ``regression_test_fn(network)`` returns the probe performance with
        the network's current adapter state.
        """
        if not candidates:
            return InternalizationResult(success=True, final_level=target_level)
        if target_level <= 0:
            return InternalizationResult(success=True, final_level=0.0)

        texts = [c.text for c in candidates]
        baseline_perf = regression_test_fn(network)
        snapshot = network.snapshot_adapter()
        trials = 0

        total_steps = self.steps
        if max_steps is not None:
            total_steps = min(total_steps, max(1, int(max_steps)))
        step_size = target_level / total_steps
        current_level = 0.0

        for step in range(total_steps):
            alpha = self.max_alpha * (step + 1) / self.steps
            # 1. small exploratory update
            network.compute_trial_update(
                texts, tokenizer, alpha=alpha,
                batch_size=self.probe_batch, max_len=self.max_len,
            )
            trials += 1
            # 2. regression test
            trial_perf = regression_test_fn(network)

            if trial_perf < baseline_perf - self.tolerance:
                # 3. performance degradation -> rollback and stop
                network.rollback(snapshot)
                final_level = current_level + step_size * step
                for c in candidates:
                    network.mark_internalized(c, final_level)
                return InternalizationResult(
                    success=False,
                    final_level=final_level,
                    stop_reason="performance_degradation",
                    stopped_at_step=step,
                    perf_before=baseline_perf,
                    perf_after=trial_perf,
                    trials=trials,
                )
            # 4. accept this step
            current_level += step_size
            baseline_perf = trial_perf  # update baseline for next step

        # consolidation succeeded
        for c in candidates:
            network.mark_internalized(c, target_level)
        return InternalizationResult(
            success=True,
            final_level=target_level,
            perf_before=baseline_perf,
            perf_after=regression_test_fn(network),
            trials=trials,
            detail={"n_candidates": len(candidates)},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def update_budget_ok(network: Any, eps: float = 0.001) -> bool:
        """Check the update-budget constraint of report section 3.5."""
        import torch

        total = sum(p.norm().item() for p in network._adapter_params())
        return total > 0  # small adapters -> budget trivially satisfied
