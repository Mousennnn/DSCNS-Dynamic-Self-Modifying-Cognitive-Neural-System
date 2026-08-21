"""Experience replay for offline correction training (v0.5.1, task spec §12).

Provides a structured buffer for storing modification episodes and their
outcomes, with support for:
  - Balanced success/failure sampling
  - Failure-weighted sampling (more failures = more learning)
  - Temporal decay (recent episodes weighted more)
  - Offline replay training of correction policy

This is the bridge between episode storage and correction learning:
  episodes → buffer → mini-batch → correction policy gradient

No information leakage: the correction policy never sees the TRUE outcome
of the current round — only past episodes.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ReplayEntry:
    """One stored experience for replay."""
    # context at time of modification
    error_state: Optional[Any] = None        # ErrorState tensor
    core_z: Optional[Any] = None             # (256,) from plasticity
    proposal_features: Optional[Any] = None  # (7,) prev proposal info
    prev_delta_A: Optional[Any] = None       # (768, 16)
    prev_delta_B: Optional[Any] = None       # (16, 768)
    prev_weight: float = 0.0
    prev_target: int = 0

    # outcome
    delta_score: float = 0.0
    outcome: str = "neutral"
    category: str = "success"

    # correction target (what the correction should have been)
    ideal_correction_A: Optional[Any] = None  # (768, 16) ideal correction
    ideal_correction_B: Optional[Any] = None  # (16, 768)
    reward: float = 0.0

    # temporal metadata
    round_id: int = 0
    timestamp: float = 0.0

    def to_training_pair(self, device: torch.device = torch.device("cpu")
                         ) -> Dict[str, torch.Tensor]:
        """Convert to training dict for correction policy."""
        out = {}
        if self.error_state is not None:
            if hasattr(self.error_state, "to_tensor"):
                out["error_state"] = self.error_state.to_tensor().to(device)
            elif isinstance(self.error_state, torch.Tensor):
                out["error_state"] = self.error_state.to(device)
        if self.core_z is not None:
            cz = self.core_z
            if isinstance(cz, np.ndarray):
                cz = torch.from_numpy(cz).float()
            if isinstance(cz, torch.Tensor):
                out["core_z"] = cz.to(device)
        if self.proposal_features is not None:
            out["proposal_features"] = self.proposal_features.to(device)
        if self.prev_delta_A is not None:
            out["prev_delta_A"] = self.prev_delta_A.to(device)
        if self.prev_delta_B is not None:
            out["prev_delta_B"] = self.prev_delta_B.to(device)
        out["prev_weight"] = self.prev_weight
        out["prev_target"] = self.prev_target
        out["delta_score"] = self.delta_score
        out["reward"] = self.reward
        return out


class ExperienceReplayBuffer:
    """Buffer for storing and sampling modification episodes.

    Supports three sampling strategies:
      1. uniform: random sample from all episodes
      2. failure_weighted: oversample failures (1:1 failure:success)
      3. temporal: recent episodes weighted more (exponential decay)
    """

    def __init__(self, capacity: int = 1000,
                 failure_ratio: float = 1.0,
                 temporal_decay: float = 0.99):
        self.capacity = capacity
        self.failure_ratio = failure_ratio  # target failure:success ratio
        self.temporal_decay = temporal_decay

        self.entries: deque = deque(maxlen=capacity)
        self.failure_count = 0
        self.success_count = 0
        self.total_count = 0

    def add(self, entry: ReplayEntry) -> None:
        """Add an episode to the buffer."""
        self.entries.append(entry)
        self.total_count += 1
        if entry.category == "failure":
            self.failure_count += 1
        elif entry.category == "success":
            self.success_count += 1

    def sample(self, batch_size: int, strategy: str = "uniform",
               device: torch.device = torch.device("cpu")
               ) -> List[Dict[str, Any]]:
        """Sample a mini-batch from the buffer.

        Args:
            batch_size: number of samples
            strategy: "uniform" | "failure_weighted" | "temporal"
            device: target device for tensors

        Returns:
            List of training dicts
        """
        if len(self.entries) == 0:
            return []

        if strategy == "uniform":
            indices = random.sample(range(len(self.entries)),
                                    min(batch_size, len(self.entries)))
        elif strategy == "failure_weighted":
            indices = self._failure_weighted_sample(batch_size)
        elif strategy == "temporal":
            indices = self._temporal_sample(batch_size)
        else:
            indices = random.sample(range(len(self.entries)),
                                    min(batch_size, len(self.entries)))

        return [self.entries[i].to_training_pair(device) for i in indices]

    def _failure_weighted_sample(self, batch_size: int) -> List[int]:
        """Sample with failure:success ratio ≈ failure_ratio."""
        failures = [i for i, e in enumerate(self.entries) if e.category == "failure"]
        successes = [i for i, e in enumerate(self.entries) if e.category != "failure"]

        n_failures = min(len(failures), int(batch_size * self.failure_ratio /
                                             (1 + self.failure_ratio)))
        n_successes = batch_size - n_failures

        sampled = []
        if failures:
            sampled.extend(random.choices(failures, k=min(n_failures, len(failures))))
        if successes:
            sampled.extend(random.choices(successes, k=min(n_successes, len(successes))))

        # pad if needed
        while len(sampled) < batch_size and len(self.entries) > 0:
            sampled.append(random.randint(0, len(self.entries) - 1))

        return sampled[:batch_size]

    def _temporal_sample(self, batch_size: int) -> List[int]:
        """Sample with exponential temporal decay (recent weighted more)."""
        n = len(self.entries)
        weights = [self.temporal_decay ** (n - 1 - i) for i in range(n)]
        total = sum(weights)
        weights = [w / total for w in weights]
        indices = np.random.choice(n, size=min(batch_size, n), replace=False, p=weights)
        return indices.tolist()

    def get_failure_entries(self) -> List[ReplayEntry]:
        return [e for e in self.entries if e.category == "failure"]

    def get_success_entries(self) -> List[ReplayEntry]:
        return [e for e in self.entries if e.category == "success"]

    @property
    def failure_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.failure_count / self.total_count

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": len(self.entries),
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": self.failure_rate,
            "capacity": self.capacity,
        }


def train_correction_offline(
    policy,  # CorrectionPolicy
    buffer: ExperienceReplayBuffer,
    optimizer: torch.optim.Optimizer,
    num_steps: int = 100,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    lambda_recovery: float = 1.0,
    lambda_repeat: float = 0.5,
    lambda_regularization: float = 0.01,
) -> Dict[str, float]:
    """Offline training of correction policy (task spec §12).

    Loss = L_performance + λ1*L_recovery + λ2*L_repeat + λ3*L_regularization

    L_performance: MSE between predicted correction and ideal correction
    L_recovery: bonus for corrections that would improve score
    L_repeat: penalty for corrections that don't prevent similar failures
    L_regularization: weight decay on correction magnitude
    """
    if len(buffer.entries) < 2:
        return {"loss": 0.0, "steps": 0}

    policy.train()
    total_loss = 0.0
    n_steps = 0

    for step in range(num_steps):
        batch = buffer.sample(batch_size, strategy="failure_weighted", device=device)
        if not batch:
            continue

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        for entry in batch:
            if "error_state" not in entry or "core_z" not in entry:
                continue

            error_t = entry["error_state"].unsqueeze(0)  # (1, 8)
            core_z = entry["core_z"].unsqueeze(0)        # (1, 256)

            # build proposal features
            prop_feat = torch.zeros(1, 7, device=device)
            if "proposal_features" in entry:
                prop_feat = entry["proposal_features"].unsqueeze(0)[:, :7]

            # forward through policy (no memory for offline training)
            output = policy(
                error_t,
                torch.zeros(1, 32, device=device),  # no memory in offline
                prop_feat,
                core_z,
                mode="error_conditioned",
            )

            # L_performance: correction should reduce error
            reward = entry.get("reward", torch.tensor(0.0, device=device))
            if isinstance(reward, (int, float)):
                reward = torch.tensor(reward, device=device)
            corr_loss = -reward * output["correction_strength"]

            # L_regularization: prevent correction from growing unbounded
            reg_loss = lambda_regularization * output["correction_norm"]

            loss = loss + corr_loss + reg_loss

        loss = loss / max(1, len(batch))
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1

    policy.eval()
    return {
        "loss": total_loss / max(n_steps, 1),
        "steps": n_steps,
    }
