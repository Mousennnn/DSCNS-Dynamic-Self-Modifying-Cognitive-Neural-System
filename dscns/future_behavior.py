"""Future Behavior Evaluation (v0.5.2).

Evaluates how *current* modifications relate to *past* experiences:

  * P(FutureModification | Experience)   — how often a modification is
    derived from / linked to a past experience.
  * Repeat Failure Rate                   — future modifications similar to
    past failures that fail again (absorption failure).
  * Successful Reuse Rate                 — future modifications similar to
    past successes that succeed again (knowledge reuse).
  * Modification similarity shift         — drift of modification directions
    relative to history over time.
  * Target / magnitude shift              — drift in what is modified.

The module distinguishes two behaviors explicitly:

  * SUCCESS reuse   — current modification is similar to a past *success*;
    the system repeats what worked.
  * FAILURE avoidance — current modification is similar to a past *failure*
    but *avoids* repeating it (different target / direction / magnitude),
    and therefore succeeds.

All similarity is cosine similarity on modification *directions*
(delta_theta), not raw norms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ====================================================================== #
# Modification similarity tracker
# ====================================================================== #

class ModificationSimilarityTracker:
    """Tracks cosine similarity between current and historical
    modification directions (delta_theta) over time.

    A modification direction is the normalized flattened delta:
    direction = flatten(delta_W_A, delta_W_B) / ||flatten(...)||.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.rounds: List[int] = []
        self.directions: List[np.ndarray] = []   # unit vectors
        self.outcomes: List[str] = []
        self.magnitudes: List[float] = []
        self.targets: List[int] = []

    # ------------------------------------------------------------------ #
    def add(self, delta_theta: Any, round_id: int = 0, outcome: str = "neutral",
            magnitude: Optional[float] = None,
            target: Optional[int] = None) -> None:
        """Record one modification direction."""
        self.rounds.append(int(round_id))
        self.directions.append(self.direction_vector(delta_theta))
        self.outcomes.append(str(outcome))
        self.magnitudes.append(float(magnitude) if magnitude is not None else 0.0)
        self.targets.append(int(target) if target is not None else -1)
        if len(self.rounds) > self.capacity:
            self.rounds = self.rounds[-self.capacity:]
            self.directions = self.directions[-self.capacity:]
            self.outcomes = self.outcomes[-self.capacity:]
            self.magnitudes = self.magnitudes[-self.capacity:]
            self.targets = self.targets[-self.capacity:]

    # ------------------------------------------------------------------ #
    def similarity(self, delta_theta: Any,
                   reference: str = "all") -> Dict[str, float]:
        """Cosine similarity of a direction vs stored history.

        reference: 'all' | 'success' | 'failure' — restrict the history.
        Returns {'max': .., 'mean': .., 'n': ..}.
        """
        q = self.direction_vector(delta_theta)
        idx = range(len(self.directions))
        if reference in ("success", "failure"):
            idx = [i for i in idx if self.outcomes[i] == reference]
        if not idx:
            return {"max": 0.0, "mean": 0.0, "n": 0}
        sims = []
        for i in idx:
            a, b = q, self.directions[i]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm < 1e-12:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(a, b) / norm))
        return {"max": float(max(sims)), "mean": float(np.mean(sims)),
                "n": len(sims)}

    def similarity_shift(self, window: int = 10) -> Dict[str, float]:
        """Mean similarity of recent vs older modifications to all history.

        ``modification_similarity_shift`` = mean_recent_similarity -
        mean_older_similarity.  A negative shift means recent modifications
        diverge from historical ones (novel directions).
        """
        n = len(self.directions)
        if n < 2:
            return {"modification_similarity_shift": 0.0,
                    "mean_recent": 0.0, "mean_older": 0.0}
        window = max(1, min(window, n // 2))
        recent = self.directions[-window:]
        older = self.directions[:-window]
        if not older:
            return {"modification_similarity_shift": 0.0,
                    "mean_recent": 0.0, "mean_older": 0.0}
        mean_recent = self._mean_pairwise(recent)
        mean_older = self._mean_pairwise(older)
        return {
            "modification_similarity_shift": mean_recent - mean_older,
            "mean_recent": mean_recent,
            "mean_older": mean_older,
        }

    def target_shift(self, window: int = 10) -> Dict[str, Any]:
        """Shift in target-group usage between older and recent rounds."""
        n = len(self.targets)
        if n < 2:
            return {"target_shift": 0.0, "old_dist": {}, "new_dist": {}}
        window = max(1, min(window, n // 2))
        old = self.targets[:-window]
        new = self.targets[-window:]
        old_dist = _distribution(old)
        new_dist = _distribution(new)
        shift = _distribution_distance(old_dist, new_dist)
        return {"target_shift": shift, "old_dist": old_dist,
                "new_dist": new_dist}

    def magnitude_shift(self, window: int = 10) -> Dict[str, Any]:
        """Relative magnitude drift: (mean_recent - mean_old) / max(mean_old, eps)."""
        n = len(self.magnitudes)
        if n < 2:
            return {"magnitude_shift": 0.0, "mean_recent": 0.0,
                    "mean_older": 0.0, "relative": 0.0}
        window = max(1, min(window, n // 2))
        old = np.asarray(self.magnitudes[:-window], dtype=np.float64)
        new = np.asarray(self.magnitudes[-window:], dtype=np.float64)
        mean_old = float(old.mean()) if old.size else 0.0
        mean_new = float(new.mean()) if new.size else 0.0
        rel = (mean_new - mean_old) / max(abs(mean_old), 1e-12)
        return {"magnitude_shift": mean_new - mean_old,
                "mean_recent": mean_new, "mean_older": mean_old,
                "relative": float(rel)}

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_modifications": len(self.rounds),
            "similarity_shift": self.similarity_shift(),
            "target_shift": self.target_shift(),
            "magnitude_shift": self.magnitude_shift(),
            "series": [
                {"round_id": r, "outcome": o, "magnitude": m, "target": t}
                for r, o, m, t in zip(self.rounds, self.outcomes,
                                      self.magnitudes, self.targets)
            ],
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def direction_vector(delta_theta: Any) -> np.ndarray:
        """Flatten delta_theta into a unit vector (direction only)."""
        if delta_theta is None:
            return np.zeros(1, dtype=np.float32)
        if isinstance(delta_theta, dict):
            parts = []
            for key in ("delta_W_A", "delta_W_B", "dA", "dB"):
                if key in delta_theta and delta_theta[key] is not None:
                    parts.append(_to_numpy(delta_theta[key]))
            if parts:
                v = np.concatenate([p.flatten() for p in parts])
            else:
                return np.zeros(1, dtype=np.float32)
        else:
            v = _to_numpy(delta_theta).flatten()
        v = v.astype(np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return np.zeros(max(v.size, 1), dtype=np.float64)
        return v / norm

    @staticmethod
    def cosine(a: Any, b: Any) -> float:
        """Cosine similarity between two modification directions."""
        va = ModificationSimilarityTracker.direction_vector(a)
        vb = ModificationSimilarityTracker.direction_vector(b)
        if va.size == 0 or vb.size == 0:
            return 0.0
        if va.size != vb.size:
            # pad the shorter to compare
            n = max(va.size, vb.size)
            va = np.pad(va, (0, n - va.size))
            vb = np.pad(vb, (0, n - vb.size))
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm < 1e-12:
            return 0.0
        return float(np.dot(va, vb) / norm)

    @staticmethod
    def _mean_pairwise(dirs: List[np.ndarray]) -> float:
        if len(dirs) < 2:
            return 0.0
        sims = []
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                a, b = dirs[i], dirs[j]
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm >= 1e-12:
                    sims.append(float(np.dot(a, b) / norm))
        return float(np.mean(sims)) if sims else 0.0


# ====================================================================== #
# Success reuse evaluator
# ====================================================================== #

class SuccessReuseEvaluator:
    """Tracks whether successful modifications get reused in similar contexts.

    Distinguishes (task spec):

      * SUCCESS reuse — a modification similar to a past *success* that
        succeeds again (reuse of what worked).
      * FAILURE avoidance — a modification similar to a past *failure*
        that succeeds because it avoided repeating it (different
        direction / target / magnitude).

    Reuse similarity is measured on modification direction (cosine) plus
    optional context similarity and target agreement.
    """

    def __init__(self, direction_weight: float = 0.5,
                 context_weight: float = 0.3,
                 target_weight: float = 0.2,
                 reuse_threshold: float = 0.5):
        self.direction_weight = direction_weight
        self.context_weight = context_weight
        self.target_weight = target_weight
        self.reuse_threshold = reuse_threshold

        # historical successes / failures with their directions
        self.successes: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

        # future modifications to evaluate
        self.future: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def record_success(self, round_id: int, delta_theta: Any,
                       context: Any = None, target: int = -1,
                       magnitude: float = 0.0) -> None:
        self.successes.append({
            "round_id": round_id,
            "direction": ModificationSimilarityTracker.direction_vector(delta_theta),
            "context": _context_vector(context),
            "target": target,
            "magnitude": magnitude,
        })

    def record_failure(self, round_id: int, delta_theta: Any,
                       context: Any = None, target: int = -1,
                       magnitude: float = 0.0) -> None:
        self.failures.append({
            "round_id": round_id,
            "direction": ModificationSimilarityTracker.direction_vector(delta_theta),
            "context": _context_vector(context),
            "target": target,
            "magnitude": magnitude,
        })

    def record_future_modification(self, round_id: int, delta_theta: Any,
                                   outcome: str, context: Any = None,
                                   target: int = -1,
                                   magnitude: float = 0.0) -> None:
        self.future.append({
            "round_id": round_id,
            "direction": ModificationSimilarityTracker.direction_vector(delta_theta),
            "context": _context_vector(context),
            "target": target,
            "magnitude": magnitude,
            "outcome": outcome,
        })

    # ------------------------------------------------------------------ #
    def _similarity_to(self, query: Dict[str, Any],
                       pool: List[Dict[str, Any]]) -> float:
        best = 0.0
        for p in pool:
            s_dir = _cos_vectors(query["direction"], p["direction"])
            s_ctx = _cos_vectors(query["context"], p["context"])
            s_tgt = 1.0 if (query["target"] >= 0 and
                            query["target"] == p["target"]) else 0.0
            sim = (self.direction_weight * s_dir +
                   self.context_weight * s_ctx +
                   self.target_weight * s_tgt)
            best = max(best, sim)
        return best

    def similarity_to_success(self, delta_theta: Any, context: Any = None,
                              target: int = -1) -> float:
        query = {
            "direction": ModificationSimilarityTracker.direction_vector(delta_theta),
            "context": _context_vector(context), "target": target,
        }
        return self._similarity_to(query, self.successes)

    def similarity_to_failure(self, delta_theta: Any, context: Any = None,
                              target: int = -1) -> float:
        query = {
            "direction": ModificationSimilarityTracker.direction_vector(delta_theta),
            "context": _context_vector(context), "target": target,
        }
        return self._similarity_to(query, self.failures)

    def is_success_reuse(self, delta_theta: Any, context: Any = None,
                         target: int = -1,
                         threshold: Optional[float] = None) -> bool:
        """Current modification is similar to a past SUCCESS (reuse)."""
        t = threshold if threshold is not None else self.reuse_threshold
        return self.similarity_to_success(delta_theta, context, target) >= t

    def is_failure_avoidance(self, delta_theta: Any, context: Any = None,
                             target: int = -1,
                             threshold: Optional[float] = None) -> bool:
        """Current modification is similar to a past FAILURE.

        Whether it *avoids* the failure is judged by its outcome: the
        caller should combine this with the actual outcome — see
        ``reuse_summary()``.
        """
        t = threshold if threshold is not None else self.reuse_threshold
        return self.similarity_to_failure(delta_theta, context, target) >= t

    # ------------------------------------------------------------------ #
    def successful_reuse_rate(self, threshold: Optional[float] = None) -> float:
        """P(success | similar to a past success) over future modifications."""
        t = threshold if threshold is not None else self.reuse_threshold
        reused = [f for f in self.future
                  if self._similarity_to(f, self.successes) >= t]
        if not reused:
            return 0.0
        return sum(1.0 for f in reused if f["outcome"] == "success") / len(reused)

    def reuse_summary(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Count the four reuse/avoidance cells over future modifications."""
        t = threshold if threshold is not None else self.reuse_threshold
        cells = {
            "reused_successes": 0,      # similar to past success AND succeeded
            "reused_failures": 0,       # similar to past success but failed
            "avoided_failures": 0,      # similar to past failure AND succeeded
            "repeated_failures": 0,     # similar to past failure AND failed
            "novel": 0,
        }
        for f in self.future:
            s_sim = self._similarity_to(f, self.successes) >= t
            f_sim = self._similarity_to(f, self.failures) >= t
            ok = f["outcome"] == "success"
            if s_sim:
                cells["reused_successes" if ok else "reused_failures"] += 1
            elif f_sim:
                cells["avoided_failures" if ok else "repeated_failures"] += 1
            else:
                cells["novel"] += 1
        total = len(self.future)
        return {
            **cells,
            "n_future": total,
            "successful_reuse_rate": (
                cells["reused_successes"] / total if total else 0.0),
            "failure_avoidance_rate": (
                cells["avoided_failures"] / total if total else 0.0),
            "repeat_failure_rate": (
                cells["repeated_failures"] / total if total else 0.0),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_successes": len(self.successes),
            "n_failures": len(self.failures),
            "n_future": len(self.future),
            "reuse_summary": self.reuse_summary(),
            "successful_reuse_rate": self.successful_reuse_rate(),
        }


# ====================================================================== #
# Future modification evaluator
# ====================================================================== #

class FutureModificationEvaluator:
    """Computes how current modifications relate to past experiences.

    Output (task spec):
      repeat_failure_rate, successful_reuse_rate,
      modification_similarity_shift, target_shift, magnitude_shift,
      p_future_given_experience.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self.similarity_tracker = ModificationSimilarityTracker()
        self.reuse_evaluator = SuccessReuseEvaluator(
            reuse_threshold=similarity_threshold)
        # (experience_id, round, outcome) history for P(FutureMod|Experience)
        self.experience_history: List[Dict[str, Any]] = []
        self.future_modifications: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def record_experience(self, round_id: int, outcome: str,
                          experience_id: str = "",
                          delta_theta: Any = None, context: Any = None,
                          target: int = -1, magnitude: float = 0.0) -> None:
        """Record a historical experience (success or failure)."""
        self.experience_history.append({
            "round_id": round_id, "outcome": outcome,
            "experience_id": experience_id,
            "delta_theta": delta_theta,
            "target": target, "magnitude": magnitude,
        })
        self.similarity_tracker.add(delta_theta, round_id, outcome,
                                    magnitude=magnitude, target=target)
        if outcome == "success":
            self.reuse_evaluator.record_success(
                round_id, delta_theta, context, target, magnitude)
        elif outcome == "failure":
            self.reuse_evaluator.record_failure(
                round_id, delta_theta, context, target, magnitude)

    def record_future_modification(self, round_id: int, delta_theta: Any,
                                   outcome: str, context: Any = None,
                                   target: int = -1, magnitude: float = 0.0,
                                   source_experience_ids: Optional[List[str]] = None,
                                   ) -> None:
        """Record a future modification (what happened *after* experience)."""
        self.future_modifications.append({
            "round_id": round_id, "delta_theta": delta_theta,
            "outcome": outcome, "context": context, "target": target,
            "magnitude": magnitude,
            "source_experience_ids": list(source_experience_ids or []),
        })
        self.reuse_evaluator.record_future_modification(
            round_id, delta_theta, outcome, context, target, magnitude)

    # ------------------------------------------------------------------ #
    def p_future_given_experience(self) -> Dict[str, float]:
        """P(FutureModification | Experience).

        Fraction of future modifications that are *derived from* a past
        experience, measured two ways:

          * 'explicit': modification records source_experience_ids
          * 'implicit': modification is similar (>= threshold) to a past
            experience direction
        """
        n = len(self.future_modifications)
        if n == 0:
            return {"explicit": 0.0, "implicit": 0.0, "n": 0}
        explicit = sum(
            1 for f in self.future_modifications if f["source_experience_ids"])
        implicit = 0
        for f in self.future_modifications:
            sim_s = self.reuse_evaluator.similarity_to_success(
                f["delta_theta"], f["context"], f["target"])
            sim_f = self.reuse_evaluator.similarity_to_failure(
                f["delta_theta"], f["context"], f["target"])
            if max(sim_s, sim_f) >= self.similarity_threshold:
                implicit += 1
        return {"explicit": explicit / n, "implicit": implicit / n, "n": n}

    def repeat_failure_rate(self) -> float:
        """P(failure | similar to a past failure) among future modifications."""
        return self.reuse_evaluator.reuse_summary()["repeat_failure_rate"]

    def successful_reuse_rate(self) -> float:
        """P(success | similar to a past success) among future modifications."""
        return self.reuse_evaluator.successful_reuse_rate()

    # ------------------------------------------------------------------ #
    def evaluate(self, window: int = 10) -> Dict[str, Any]:
        """Full evaluation report (task-spec outputs)."""
        shift = self.similarity_tracker.similarity_shift(window)
        tgt = self.similarity_tracker.target_shift(window)
        mag = self.similarity_tracker.magnitude_shift(window)
        p = self.p_future_given_experience()
        reuse = self.reuse_evaluator.reuse_summary()
        return {
            "p_future_given_experience": p,
            "repeat_failure_rate": reuse["repeat_failure_rate"],
            "successful_reuse_rate": reuse["successful_reuse_rate"],
            "failure_avoidance_rate": reuse["failure_avoidance_rate"],
            "modification_similarity_shift": shift["modification_similarity_shift"],
            "target_shift": tgt["target_shift"],
            "magnitude_shift": mag["magnitude_shift"],
            "reuse_cells": {k: v for k, v in reuse.items()
                            if k in ("reused_successes", "reused_failures",
                                     "avoided_failures", "repeated_failures",
                                     "novel")},
            "n_experiences": len(self.experience_history),
            "n_future_modifications": len(self.future_modifications),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation": self.evaluate(),
            "similarity_tracker": self.similarity_tracker.to_dict(),
            "reuse_evaluator": self.reuse_evaluator.to_dict(),
        }


# ====================================================================== #
# helpers
# ====================================================================== #

def _to_numpy(v: Any) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().float().numpy()
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "to_tensor"):
        return _to_numpy(v.to_tensor())
    return np.asarray(v, dtype=np.float64)


def _context_vector(context: Any) -> np.ndarray:
    if context is None:
        return np.zeros(1, dtype=np.float64)
    if hasattr(context, "to_tensor"):
        context = context.to_tensor()
    arr = _to_numpy(context).astype(np.float64).flatten()
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm >= 1e-12 else arr


def _cos_vectors(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        n = max(a.size, b.size)
        a = np.pad(a, (0, n - a.size))
        b = np.pad(b, (0, n - b.size))
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(np.dot(a, b) / norm)


def _distribution(values: Sequence[int]) -> Dict[int, float]:
    if not values:
        return {}
    counts: Dict[int, int] = {}
    for v in values:
        counts[int(v)] = counts.get(int(v), 0) + 1
    n = len(values)
    return {k: c / n for k, c in sorted(counts.items())}


def _distribution_distance(a: Dict[int, float],
                           b: Dict[int, float]) -> float:
    """Total-variation distance between two target distributions."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
