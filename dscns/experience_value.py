"""Experience Value Model (v0.5.3 / Phase 5.5).

Each experience E_i has a value V(E_i) that determines how much it
influences future modification decisions.

    V_i = Reward_i × Confidence_i

Where:
  - Reward_i = cumulative credit (from ExperienceCreditAssigner)
  - Confidence_i = f(n_verifications, n_successes, stability)

Value is NOT immutable — it updates over time via:
    V_{i,t+1} = V_{i,t} + α × (Target_i - V_{i,t})

This creates a learning signal: experiences that repeatedly succeed
gain value, while experiences that fail lose value.

Components:
  ExperienceValue       -- single experience value record
  ExperienceValueModel  -- manages values for all experiences
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExperienceValue:
    """Value record for a single experience.

    V = reward × confidence
    Where confidence grows with repeated verification.
    """
    experience_id: str = ""
    round_id: int = 0
    # core value components
    reward: float = 0.0                # cumulative credit from credit assigner
    confidence: float = 0.5            # reliability confidence [0, 1]
    value: float = 0.0                 # V = reward × confidence
    # tracking
    n_verifications: int = 0           # how many times this experience was verified
    n_successes: int = 0               # how many verifications were successes
    n_failures: int = 0                # how many verifications were failures
    # outcome
    experience_type: str = "failure"   # failure / success / recovery
    target_group: int = 0
    magnitude: float = 0.0
    direction_hash: str = ""           # hash of modification direction for lookup
    # update history
    value_history: List[float] = field(default_factory=list)
    # staleness
    last_used_round: int = 0
    age: int = 0                       # rounds since creation

    def update_confidence(self) -> None:
        """Update confidence based on verification history.

        Confidence = n_successes / (n_verifications + 1)
        Clamped to [0.1, 1.0] to avoid zero-confidence collapse.
        """
        total = self.n_verifications
        if total == 0:
            self.confidence = 0.5
        else:
            self.confidence = float(np.clip(
                self.n_successes / max(total, 1), 0.1, 1.0))

    def compute_value(self) -> float:
        """V = reward × confidence."""
        self.value = self.reward * self.confidence
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "round_id": self.round_id,
            "reward": self.reward,
            "confidence": self.confidence,
            "value": self.value,
            "n_verifications": self.n_verifications,
            "n_successes": self.n_successes,
            "n_failures": self.n_failures,
            "experience_type": self.experience_type,
            "target_group": self.target_group,
            "magnitude": self.magnitude,
            "age": self.age,
            "last_used_round": self.last_used_round,
        }


class ExperienceValueModel:
    """Manages experience values for all recorded experiences.

    Core operations:
      1. record(experience) → assign initial value
      2. verify(experience_id, success) → update confidence + value
      3. get_value(experience_id) → current value
      4. rank_by_value(experience_ids) → sorted by value
      5. update_all() → age all experiences, decay stale values

    Design:
      - Failure experiences start with negative value
      - Success experiences start with positive value
      - Value is updated through verification (repeated success → higher value)
      - Stale experiences decay toward zero (prevents ancient experiences
        from dominating decisions)
    """

    def __init__(self, learning_rate: float = 0.1,
                 decay_rate: float = 0.001,
                 min_confidence: float = 0.1,
                 max_confidence: float = 1.0,
                 capacity: int = 5000):
        self.learning_rate = float(learning_rate)
        self.decay_rate = float(decay_rate)
        self.min_confidence = float(min_confidence)
        self.max_confidence = float(max_confidence)
        self.capacity = capacity

        self.values: Dict[str, ExperienceValue] = {}
        self.history: List[Dict[str, Any]] = []

    def record(self, experience_id: str, round_id: int,
               reward: float = 0.0,
               experience_type: str = "failure",
               target_group: int = 0,
               magnitude: float = 0.0,
               direction_hash: str = "") -> ExperienceValue:
        """Record a new experience and assign initial value.

        Args:
            experience_id: unique identifier.
            round_id: the round this experience occurred.
            reward: initial reward (cumulative credit).
            experience_type: failure/success/recovery.
            target_group: which parameter group was modified.
            magnitude: modification magnitude.
            direction_hash: hash of the modification direction.

        Returns:
            The newly created ExperienceValue.
        """
        ev = ExperienceValue(
            experience_id=experience_id,
            round_id=round_id,
            reward=reward,
            experience_type=experience_type,
            target_group=target_group,
            magnitude=magnitude,
            direction_hash=direction_hash,
            last_used_round=round_id,
        )
        # initial confidence based on type
        if experience_type == "success":
            ev.confidence = 0.7  # success starts with moderate confidence
        elif experience_type == "failure":
            ev.confidence = 0.3  # failure starts with low confidence
        else:
            ev.confidence = 0.5

        ev.compute_value()
        ev.value_history.append(ev.value)
        self.values[experience_id] = ev

        # enforce capacity
        if len(self.values) > self.capacity:
            self._evict_oldest()

        return ev

    def verify(self, experience_id: str, success: bool) -> Optional[ExperienceValue]:
        """Verify an experience (repeated observation).

        When the system encounters a similar context and gets a result,
        the original experience's confidence is updated.

        Args:
            experience_id: the experience to verify.
            success: whether the verification was a success.

        Returns:
            Updated ExperienceValue, or None if not found.
        """
        ev = self.values.get(experience_id)
        if ev is None:
            return None

        ev.n_verifications += 1
        if success:
            ev.n_successes += 1
        else:
            ev.n_failures += 1

        ev.update_confidence()
        ev.compute_value()
        ev.value_history.append(ev.value)
        return ev

    def update_value(self, experience_id: str,
                     new_reward: Optional[float] = None) -> Optional[ExperienceValue]:
        """Update an experience's value using temporal difference learning.

        V_{t+1} = V_t + α × (Target - V_t)

        Where Target = new_reward + γ × V_future (if available).

        Args:
            experience_id: the experience to update.
            new_reward: the new reward target (if None, only age update).

        Returns:
            Updated ExperienceValue, or None if not found.
        """
        ev = self.values.get(experience_id)
        if ev is None:
            return None

        if new_reward is not None:
            target = new_reward
            ev.reward = new_reward
            ev.value = ev.value + self.learning_rate * (target - ev.value)
        else:
            # just age-based decay
            ev.value *= (1.0 - self.decay_rate)

        ev.value_history.append(ev.value)
        return ev

    def get_value(self, experience_id: str) -> float:
        """Get current value of an experience."""
        ev = self.values.get(experience_id)
        if ev is None:
            return 0.0
        return ev.value

    def rank_by_value(self, experience_ids: List[str],
                      descending: bool = True) -> List[str]:
        """Rank experience IDs by their current value."""
        scored = [(eid, self.get_value(eid)) for eid in experience_ids]
        scored.sort(key=lambda x: x[1], reverse=descending)
        return [eid for eid, _ in scored]

    def rank_by_type_and_value(self, experience_type: str,
                               descending: bool = True) -> List[str]:
        """Rank experiences of a given type by value."""
        filtered = [(eid, ev) for eid, ev in self.values.items()
                    if ev.experience_type == experience_type]
        filtered.sort(key=lambda x: x[1].value, reverse=descending)
        return [eid for eid, _ in filtered]

    def get_failure_values(self) -> List[ExperienceValue]:
        """Get all failure experiences sorted by value (lowest first)."""
        failures = [ev for ev in self.values.values()
                    if ev.experience_type == "failure"]
        failures.sort(key=lambda x: x.value)
        return failures

    def get_success_values(self) -> List[ExperienceValue]:
        """Get all success experiences sorted by value (highest first)."""
        successes = [ev for ev in self.values.values()
                     if ev.experience_type == "success"]
        successes.sort(key=lambda x: x.value, reverse=True)
        return successes

    def age_all(self) -> None:
        """Age all experiences (increment age, apply decay)."""
        for ev in self.values.values():
            ev.age += 1
            # decay based on age and staleness
            rounds_since_use = ev.age - (ev.age - ev.last_used_round)
            ev.value *= (1.0 - self.decay_rate * ev.age / 1000.0)

    def similarity_group_values(self, direction_hashes: List[str]
                                ) -> Dict[str, float]:
        """Get mean value for a group of direction hashes."""
        values = []
        for dh in direction_hashes:
            for ev in self.values.values():
                if ev.direction_hash == dh:
                    values.append(ev.value)
                    break
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        arr = np.array(values)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "n": len(values)}

    def summary(self) -> Dict[str, Any]:
        """Summary statistics over all experience values."""
        if not self.values:
            return {"n_experiences": 0}
        all_vals = [ev.value for ev in self.values.values()]
        by_type: Dict[str, List[float]] = {}
        for ev in self.values.values():
            by_type.setdefault(ev.experience_type, []).append(ev.value)
        arr = np.array(all_vals)
        type_stats = {}
        for t, vals in by_type.items():
            v = np.array(vals)
            type_stats[t] = {"mean": float(v.mean()), "std": float(v.std()),
                             "n": len(vals)}
        return {
            "n_experiences": len(self.values),
            "mean_value": float(arr.mean()),
            "std_value": float(arr.std()),
            "by_type": type_stats,
        }

    def _evict_oldest(self) -> None:
        """Remove the oldest, lowest-value experiences to fit capacity."""
        if len(self.values) <= self.capacity:
            return
        # sort by age (oldest first), then by value (lowest first)
        sorted_evs = sorted(self.values.items(),
                            key=lambda x: (-x[1].age, x[1].value))
        n_to_remove = len(self.values) - self.capacity
        for i in range(n_to_remove):
            eid = sorted_evs[i][0]
            del self.values[eid]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "values": {eid: ev.to_dict() for eid, ev in self.values.items()},
        }
