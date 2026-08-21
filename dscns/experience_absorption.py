"""Experience Absorption Tracking (v0.5.2).

Tracks how past experiences — especially failures — influence future
modifications.  A system that genuinely learns from experience should show:

  1. *Lineage*: each failure is explicitly linked to the future
     modifications that used it (source_experience_ids), so we can ask
     "did this failure lead to a later success?".

  2. *Lower future repeat-failure rate*: the fraction of modifications
     that repeat a past failure should drop as experience accumulates.

  3. *Positive Experience Absorption Rate*:

         EAR = 1 - RFR_future / RFR_baseline

     EAR > 0 means the future repeat-failure rate is lower than the
     baseline repeat-failure rate, i.e. failures are being absorbed.

This module is a model-side *measurement + bookkeeping* component; it
does not implement the modification policy itself.  It integrates with
the v0.5.1 episodic memory (modification_memory.EpisodicModificationRecord
/ memory_encoder.ModificationEpisode) by accepting the same field names
(round_id, context, error, proposal, target, magnitude, delta_theta,
outcome, memory_ids, memory_similarity, correction, weight_before,
weight_after, future_similarity, future_outcome).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ====================================================================== #
# Experience record + lineage
# ====================================================================== #

@dataclass
class ExperienceRecord:
    """One recorded experience (failure or future modification).

    Every failure gets a unique ``experience_id``; future modifications
    that drew on it record the failure id in ``source_experience_ids``.
    """
    round_id: int = 0
    context: Any = None              # context embedding (context_dim,)
    error: Any = None                # ErrorState / error tensor
    proposal: Any = None             # dict {delta_W_A, delta_W_B} or tensor
    target: int = 0                  # target group
    magnitude: float = 0.0           # applied magnitude (weight)
    delta_theta: Any = None          # dict {delta_W_A, delta_W_B} or tensor
    outcome: str = "neutral"         # success / failure / recovery / ...
    memory_ids: List[str] = field(default_factory=list)
    memory_similarity: float = 0.0   # max similarity to retrieved memories
    correction: Any = None           # dict or float (correction strength)
    weight_before: float = 0.0
    weight_after: float = 0.0
    future_similarity: float = 0.0   # similarity of a later mod. to this one
    future_outcome: str = ""         # outcome of the linked later modification

    # bookkeeping
    experience_id: str = ""
    source_experience_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_failure(self) -> bool:
        return self.outcome == "failure"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k in ("context", "error", "proposal", "delta_theta", "correction"):
                d[k] = _serialize(v)
            else:
                d[k] = v.tolist() if hasattr(v, "tolist") else v
        return d


@dataclass
class ExperienceLineage:
    """Links one failure experience to the future modifications that used it.

    ``efficacy`` = fraction of linked future modifications that succeeded,
    i.e. the failure was actually absorbed (turned into a future success).
    """
    failure_experience_id: str = ""
    failure_round_id: int = -1
    future_round_ids: List[int] = field(default_factory=list)
    future_modification_ids: List[str] = field(default_factory=list)
    future_outcomes: List[str] = field(default_factory=list)
    future_similarities: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def n_uses(self) -> int:
        return len(self.future_round_ids)

    @property
    def n_successes(self) -> int:
        return sum(1 for o in self.future_outcomes if o == "success")

    @property
    def efficacy(self) -> float:
        """Fraction of linked future modifications that succeeded."""
        if self.n_uses == 0:
            return 0.0
        return self.n_successes / self.n_uses

    def add_future(self, modification_id: str, round_id: int,
                   outcome: str, similarity: float = 0.0) -> None:
        self.future_modification_ids.append(modification_id)
        self.future_round_ids.append(round_id)
        self.future_outcomes.append(outcome)
        self.future_similarities.append(float(similarity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_experience_id": self.failure_experience_id,
            "failure_round_id": self.failure_round_id,
            "future_round_ids": self.future_round_ids,
            "future_modification_ids": self.future_modification_ids,
            "future_outcomes": self.future_outcomes,
            "future_similarities": self.future_similarities,
            "n_uses": self.n_uses,
            "n_successes": self.n_successes,
            "efficacy": self.efficacy,
        }


# ====================================================================== #
# Experience tracker
# ====================================================================== #

class ExperienceTracker:
    """Records experiences and tracks failure -> future-success lineage.

    Usage::

        tracker = ExperienceTracker()
        fid = tracker.record_failure(round_id=3, target=1, error=err,
                                     proposal=prop, delta_theta=dt,
                                     outcome="failure")
        mid = tracker.record_modification(round_id=8, source_experience_ids=[fid],
                                          outcome="success", future_similarity=0.72)
        lineage = tracker.lineage_for(fid)   # failure 3 -> round 8 success
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.experiences: List[ExperienceRecord] = []
        self.lineages: Dict[str, ExperienceLineage] = {}
        self._counter = 0

    # ------------------------------------------------------------------ #
    # record APIs
    # ------------------------------------------------------------------ #
    def _next_id(self) -> str:
        self._counter += 1
        return f"exp-{self._counter:05d}"

    def record(self, *, outcome: str = "neutral",
               round_id: int = 0, context: Any = None, error: Any = None,
               proposal: Any = None, target: int = 0, magnitude: float = 0.0,
               delta_theta: Any = None, memory_ids: Optional[List[str]] = None,
               memory_similarity: float = 0.0, correction: Any = None,
               weight_before: float = 0.0, weight_after: float = 0.0,
               future_similarity: float = 0.0, future_outcome: str = "",
               source_experience_ids: Optional[List[str]] = None,
               ) -> str:
        """Record a generic experience. Returns its experience_id.

        ``outcome`` should be one of success / failure / recovery.
        """
        exp = ExperienceRecord(
            round_id=round_id, context=context, error=error,
            proposal=proposal, target=target, magnitude=magnitude,
            delta_theta=delta_theta, outcome=outcome,
            memory_ids=list(memory_ids or []),
            memory_similarity=memory_similarity, correction=correction,
            weight_before=weight_before, weight_after=weight_after,
            future_similarity=future_similarity, future_outcome=future_outcome,
            experience_id=self._next_id(),
            source_experience_ids=list(source_experience_ids or []),
        )
        self.experiences.append(exp)
        if len(self.experiences) > self.capacity:
            self.experiences = self.experiences[-self.capacity:]
        # auto-link lineage: every referenced source failure gets a link
        for sid in exp.source_experience_ids:
            self.track_lineage(failure_id=sid, modification_id=exp.experience_id,
                               future_round_id=exp.round_id,
                               future_outcome=exp.outcome,
                               future_similarity=exp.future_similarity)
        return exp.experience_id

    def record_failure(self, *, round_id: int = 0, context: Any = None,
                       error: Any = None, proposal: Any = None,
                       target: int = 0, magnitude: float = 0.0,
                       delta_theta: Any = None,
                       memory_ids: Optional[List[str]] = None,
                       memory_similarity: float = 0.0,
                       correction: Any = None,
                       weight_before: float = 0.0, weight_after: float = 0.0,
                       future_similarity: float = 0.0,
                       future_outcome: str = "",
                       ) -> str:
        """Record a failure. Each failure gets a unique experience_id."""
        return self.record(
            round_id=round_id, context=context, error=error, proposal=proposal,
            target=target, magnitude=magnitude, delta_theta=delta_theta,
            outcome="failure", memory_ids=memory_ids,
            memory_similarity=memory_similarity, correction=correction,
            weight_before=weight_before, weight_after=weight_after,
            future_similarity=future_similarity, future_outcome=future_outcome,
        )

    def record_modification(self, *, round_id: int = 0, context: Any = None,
                            error: Any = None, proposal: Any = None,
                            target: int = 0, magnitude: float = 0.0,
                            delta_theta: Any = None, outcome: str = "neutral",
                            source_experience_ids: Optional[List[str]] = None,
                            memory_ids: Optional[List[str]] = None,
                            memory_similarity: float = 0.0,
                            correction: Any = None,
                            weight_before: float = 0.0,
                            weight_after: float = 0.0,
                            future_similarity: float = 0.0,
                            future_outcome: str = "",
                            ) -> str:
        """Record a future modification, optionally linked to past failures."""
        return self.record(
            round_id=round_id, context=context, error=error, proposal=proposal,
            target=target, magnitude=magnitude, delta_theta=delta_theta,
            outcome=outcome, memory_ids=memory_ids,
            memory_similarity=memory_similarity, correction=correction,
            weight_before=weight_before, weight_after=weight_after,
            future_similarity=future_similarity, future_outcome=future_outcome,
            source_experience_ids=source_experience_ids,
        )

    # ------------------------------------------------------------------ #
    # lineage APIs
    # ------------------------------------------------------------------ #
    def track_lineage(self, failure_id: str, modification_id: str,
                      future_round_id: int, future_outcome: str,
                      future_similarity: float = 0.0) -> ExperienceLineage:
        """Link a failure to a future modification that used it."""
        failure = self.get(failure_id)
        if failure is None:
            # record the link anyway; round id may be unknown
            failure_round = -1
        else:
            failure_round = failure.round_id
        lineage = self.lineages.get(failure_id)
        if lineage is None:
            lineage = ExperienceLineage(
                failure_experience_id=failure_id,
                failure_round_id=failure_round,
            )
            self.lineages[failure_id] = lineage
        lineage.add_future(modification_id, future_round_id,
                           future_outcome, future_similarity)
        return lineage

    def lineage_for(self, failure_id: str) -> Optional[ExperienceLineage]:
        return self.lineages.get(failure_id)

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def get(self, experience_id: str) -> Optional[ExperienceRecord]:
        for exp in reversed(self.experiences):
            if exp.experience_id == experience_id:
                return exp
        return None

    def failures(self) -> List[ExperienceRecord]:
        return [e for e in self.experiences if e.outcome == "failure"]

    def successes(self) -> List[ExperienceRecord]:
        return [e for e in self.experiences if e.outcome == "success"]

    def modifications(self) -> List[ExperienceRecord]:
        """Future modifications (any non-failure recording with source links)."""
        return [e for e in self.experiences
                if e.source_experience_ids or e.outcome != "failure"]

    def linked_modifications(self) -> List[ExperienceRecord]:
        return [e for e in self.experiences if e.source_experience_ids]

    # ------------------------------------------------------------------ #
    # absorption statistics
    # ------------------------------------------------------------------ #
    def absorption_rate(self) -> float:
        """Fraction of failures that have at least one linked future success."""
        failures = self.failures()
        if not failures:
            return 0.0
        absorbed = 0
        for f in failures:
            lin = self.lineages.get(f.experience_id)
            if lin is not None and lin.n_successes > 0:
                absorbed += 1
        return absorbed / len(failures)

    def lineage_efficacy_stats(self) -> Dict[str, float]:
        """Mean / std efficacy over all recorded lineages."""
        efficacies = [lin.efficacy for lin in self.lineages.values()
                      if lin.n_uses > 0]
        if not efficacies:
            return {"mean": 0.0, "std": 0.0, "n_lineages": 0}
        arr = np.asarray(efficacies, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "n_lineages": len(efficacies)}

    def summary(self) -> Dict[str, Any]:
        return {
            "n_experiences": len(self.experiences),
            "n_failures": len(self.failures()),
            "n_successes": len(self.successes()),
            "n_lineages": len(self.lineages),
            "absorption_rate": self.absorption_rate(),
            "lineage_efficacy": self.lineage_efficacy_stats(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.summary(),
            "experiences": [e.to_dict() for e in self.experiences],
            "lineages": [lin.to_dict() for lin in self.lineages.values()],
        }


# ====================================================================== #
# Absorption evaluator (EAR)
# ====================================================================== #

class AbsorptionEvaluator:
    """Evaluates whether experiences are being absorbed.

    Core metric (task spec):

        EAR = 1 - RFR_future / RFR_baseline

    * RFR_baseline: repeat-failure rate *before* absorption (early rounds
      or the no-memory control condition).
    * RFR_future:   repeat-failure rate *after* absorption (later rounds
      or the memory condition).

    EAR > 0  -> failures are being absorbed (future repeats drop).
    EAR = 0  -> no change.
    EAR < 0  -> the system is repeating failures *more* over time.

    The RFR_baseline == 0 case is handled gracefully: no division error,
    and the result is reported as neutral (0.0) with ``baseline_zero=True``
    unless RFR_future > 0 (then EAR is clamped to -1.0).
    """

    def __init__(self, clamp: bool = True):
        self.clamp = clamp

    # ------------------------------------------------------------------ #
    def evaluate(self, rfr_baseline: float, rfr_future: float) -> Dict[str, Any]:
        """EAR = 1 - RFR_future / RFR_baseline (baseline-0 safe)."""
        rfr_baseline = float(rfr_baseline)
        rfr_future = float(rfr_future)
        baseline_zero = rfr_baseline <= 0.0
        ear: float
        note: str
        if baseline_zero:
            if rfr_future <= 0.0:
                # no baseline failures and no future failures: no signal
                ear = 0.0
                note = ("RFR_baseline == 0 and RFR_future == 0; "
                        "no absorption signal (both rates zero)")
            else:
                # failures appeared from a zero-failure baseline: worst case
                ear = -1.0
                note = ("RFR_baseline == 0 but RFR_future > 0; "
                        "repeat failures emerged from a clean baseline "
                        "(EAR clamped to -1.0)")
        else:
            ear = 1.0 - rfr_future / rfr_baseline
            note = ""
            if ear > 0:
                note = "absorption happening (future failures reduced)"
            elif ear < 0:
                note = "absorption failing (future failures increased)"
            else:
                note = "no change in repeat-failure rate"
        if self.clamp:
            ear = float(np.clip(ear, -1.0, 1.0))
        return {
            "ear": ear,
            "rfr_baseline": rfr_baseline,
            "rfr_future": rfr_future,
            "absorbed": ear > 0.0,
            "baseline_zero": baseline_zero,
            "note": note,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_rfr(records: Sequence[Any],
                    same_target: bool = True,
                    similarity_threshold: float = 0.5,
                    error_similarity: bool = True,
                    window: Optional[int] = None) -> float:
        """Repeat Failure Rate over a sequence of records.

        A failure at position t is a *repeat* if a previous failure within
        the window matched it on target (same_target) and/or error
        similarity (error_similarity).  RFR = repeats / (failures - 1)
        (the first failure has no predecessor), 0.0 if < 2 failures.

        ``records`` items may be ExperienceRecord, dicts, or objects with
        .outcome / .target (or .target_group) / .error attributes.
        """
        recs = list(records)
        if window is not None:
            recs = recs[-window:]
        failures = [r for r in recs if _outcome_of(r) == "failure"]
        if len(failures) < 2:
            return 0.0
        repeated = 0
        for i in range(1, len(failures)):
            curr, prev = failures[i], failures[i - 1]
            if same_target and _target_of(curr) == _target_of(prev):
                repeated += 1
                continue
            if error_similarity:
                sim = _error_similarity(_error_of(curr), _error_of(prev))
                if sim >= similarity_threshold:
                    repeated += 1
        return repeated / (len(failures) - 1)

    # ------------------------------------------------------------------ #
    def evaluate_from_tracker(self, tracker: ExperienceTracker,
                              baseline_window: Optional[int] = None,
                              future_window: Optional[int] = None,
                              same_target: bool = True,
                              similarity_threshold: float = 0.5,
                              error_similarity: bool = True,
                              ) -> Dict[str, Any]:
        """Evaluate EAR from a tracker's history.

        The baseline RFR is computed on the first ``baseline_window``
        experiences (early rounds, pre-absorption), and RFR_future on the
        last ``future_window`` experiences (post-absorption).
        """
        recs = tracker.experiences
        if baseline_window is not None:
            baseline_recs = recs[:baseline_window]
        else:
            # default: first half is the baseline phase
            baseline_recs = recs[:len(recs) // 2] if recs else []
        if future_window is not None:
            future_recs = recs[-future_window:] if recs else []
        else:
            future_recs = recs[len(recs) // 2:] if recs else []

        rfr_base = self.compute_rfr(baseline_recs, same_target=same_target,
                                    similarity_threshold=similarity_threshold,
                                    error_similarity=error_similarity)
        rfr_fut = self.compute_rfr(future_recs, same_target=same_target,
                                   similarity_threshold=similarity_threshold,
                                   error_similarity=error_similarity)
        result = self.evaluate(rfr_base, rfr_fut)
        result["rfr_baseline_computed"] = rfr_base
        result["rfr_future_computed"] = rfr_fut
        return result


# ====================================================================== #
# Similarity-outcome tracker
# ====================================================================== #

class SimilarityOutcomeTracker:
    """Tracks modification similarity to historical failures over time.

    Each round records how similar the current modification was to the
    *historical failure set* (0..1) together with the round's outcome.
    This exposes whether similarity to past failures predicts outcomes:
    if absorption works, modifications highly similar to past failures
    should increasingly *succeed* (failure avoidance), not repeat them.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.rounds: List[int] = []
        self.similarities: List[float] = []
        self.outcomes: List[str] = []

    def update(self, round_id: int, similarity: float, outcome: str) -> None:
        """Record (round, similarity_to_historical_failures, outcome)."""
        self.rounds.append(int(round_id))
        self.similarities.append(float(similarity))
        self.outcomes.append(str(outcome))
        if len(self.rounds) > self.capacity:
            self.rounds = self.rounds[-self.capacity:]
            self.similarities = self.similarities[-self.capacity:]
            self.outcomes = self.outcomes[-self.capacity:]

    # ------------------------------------------------------------------ #
    def series(self) -> List[Dict[str, Any]]:
        return [
            {"round_id": r, "similarity": s, "outcome": o}
            for r, s, o in zip(self.rounds, self.similarities, self.outcomes)
        ]

    def __len__(self) -> int:
        return len(self.rounds)

    # ------------------------------------------------------------------ #
    def correlation(self) -> float:
        """Pearson correlation(similarity, outcome_success)."""
        if len(self.rounds) < 2:
            return 0.0
        sim = np.asarray(self.similarities, dtype=np.float64)
        success = np.asarray(
            [1.0 if o == "success" else 0.0 for o in self.outcomes],
            dtype=np.float64)
        if sim.std() < 1e-12 or success.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(sim, success)[0, 1])

    def mean_similarity_by_outcome(self) -> Dict[str, float]:
        by: Dict[str, List[float]] = {}
        for s, o in zip(self.similarities, self.outcomes):
            by.setdefault(o, []).append(s)
        return {o: float(np.mean(v)) for o, v in by.items()}

    def high_similarity_success_rate(self, threshold: float = 0.5) -> float:
        """Success rate among rounds whose similarity to past failures >= t."""
        high = [(s, o) for s, o in zip(self.similarities, self.outcomes)
                if s >= threshold]
        if not high:
            return 0.0
        return sum(1.0 for _, o in high if o == "success") / len(high)

    def high_similarity_failure_rate(self, threshold: float = 0.5) -> float:
        """Failure rate among rounds similar to past failures (repeats)."""
        high = [(s, o) for s, o in zip(self.similarities, self.outcomes)
                if s >= threshold]
        if not high:
            return 0.0
        return sum(1.0 for _, o in high if o == "failure") / len(high)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rounds": len(self.rounds),
            "correlation": self.correlation(),
            "mean_similarity_by_outcome": self.mean_similarity_by_outcome(),
            "high_similarity_success_rate": self.high_similarity_success_rate(),
            "high_similarity_failure_rate": self.high_similarity_failure_rate(),
            "series": self.series(),
        }


# ====================================================================== #
# shared helpers
# ====================================================================== #

def _serialize(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().float().numpy().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _serialize(x) for k, x in v.items()}
    if hasattr(v, "to_tensor"):
        return _serialize(v.to_tensor())
    if hasattr(v, "to_dict"):
        return v.to_dict()
    return v


def _outcome_of(r: Any) -> str:
    if isinstance(r, dict):
        return r.get("outcome", "neutral")
    return str(getattr(r, "outcome", "neutral"))


def _target_of(r: Any) -> int:
    if isinstance(r, dict):
        return int(r.get("target", r.get("target_group", -1)))
    t = getattr(r, "target", getattr(r, "target_group", -1))
    try:
        return int(t)
    except (TypeError, ValueError):
        return -1


def _error_of(r: Any) -> Any:
    if isinstance(r, dict):
        return r.get("error")
    return getattr(r, "error", getattr(r, "error_state", None))


def _error_similarity(a: Any, b: Any) -> float:
    """Cosine similarity between two error representations (None-safe)."""
    if a is None or b is None:
        return 0.0
    if hasattr(a, "to_tensor"):
        a = a.to_tensor()
    if hasattr(b, "to_tensor"):
        b = b.to_tensor()
    try:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a = a.detach().cpu().float().flatten()
            b = b.detach().cpu().float().flatten()
            norm = a.norm() * b.norm()
            if norm < 1e-12:
                return 0.0
            return float(torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
        a_np = np.asarray(a, dtype=np.float64).flatten()
        b_np = np.asarray(b, dtype=np.float64).flatten()
        if a_np.size == 0 or b_np.size == 0:
            return 0.0
        norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if norm < 1e-12:
            return 0.0
        return float(np.dot(a_np, b_np) / norm)
    except (TypeError, ValueError):
        return 0.0
