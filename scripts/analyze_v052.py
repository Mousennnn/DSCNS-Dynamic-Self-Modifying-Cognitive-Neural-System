"""v0.5.2 analysis: 12 long-term curves + 4-phase analysis + statistics +
experience-absorption analysis for the Phase 5.2 v0.5.2 experiments.

Produced artifacts (input root ``experiments/phase5_2_v052``):

figures/          12 long-term curves (c01..c12), phase comparison,
                  EAR / lineage / success-reuse absorption figures.
statistics/       statistical_report.md, statistical_results.json,
                  per_seed_breakdown.csv/.md, phase_analysis.csv/.md,
                  absorption_summary.json.

The script is defensive: it reads whatever is available.  A condition that
has not been run (or has no round data) is skipped with a warning; every
figure is emitted only when its data source exists, falling back to proxy
metrics otherwise.

Run:
    python scripts/analyze_v052.py [--input experiments/phase5_2_v052]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================================================================== #
# Constants                                                               #
# ====================================================================== #

DEFAULT_INPUT = os.path.join("experiments", "phase5_2_v052")
FULL_CONDITION = "Full"

EXPERIMENT_CONDITIONS = [
    "Full", "NoMemory", "NoReplay", "NoDirection", "NoOutcome",
    "PureReversal", "ErrorOnly", "RandomMemory", "ZeroMemory",
]

# 4 phases from docs/PHASE5_2_v052.md §25 (phase_boundaries: [50,150,300,450])
DEFAULT_PHASES = [(1, 50), (51, 150), (151, 300), (301, 450)]
PHASE_NAMES = ["Early Adaptation", "Experience Accumulation",
               "Experience Reuse", "Long-term Stability"]

RUNNING_WINDOW = 50          # default window for "running" curves
EAR_WINDOW = 50              # window for the running EAR curve
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 1234
RFR_WINDOW = 50              # window for running RFR computation

# Candidate JSON field names (v0.5.1 log names first, then v0.5.2 names)
KEY_ROUND = ("round", "round_id")
KEY_WEIGHT = ("magnitude_applied", "magnitude", "weight")
KEY_TARGET = ("target_group", "target")
KEY_APPLIED = ("applied_change", "delta_norm")
KEY_PROBE = ("probe_delta", "delta_score")
KEY_MEM_SIM = ("memory_similarity", "retrieval_similarity", "retrieval_sim",
               "similarity", "memory_retrieval_similarity")
KEY_FUT_SIM = ("future_similarity", "modification_similarity",
               "sim_to_history", "future_modification_similarity")
KEY_CORR_SIM = ("correction_direction_similarity", "correction_similarity",
                "corr_dir_sim", "correction_alignment")
KEY_MEM_AGE = ("memory_age", "retrieval_age")
KEY_EXP_ID = ("experience_id", "exp_id")
KEY_SOURCE = ("source_experience_ids", "source_ids")

SUCCESS_OUTCOMES = ("success", "partial_success")


# ====================================================================== #
# Small statistics helpers (scipy preferred, pure-numpy fallback)        #
# ====================================================================== #

try:  # pragma: no cover - environment dependent
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    HAVE_SCIPY = False


def _normal_p(z: float) -> float:
    """Two-tailed p-value for a z-score under the normal approximation."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _rankdata(vals: np.ndarray) -> np.ndarray:
    """Tie-aware rankdata (1-based, average ranks for ties)."""
    vals = np.asarray(vals, dtype=float)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(vals)
    i = 0
    n = len(vals)
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def ttest_paired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired t-test -> (t, p).  Uses scipy when available."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    if HAVE_SCIPY:
        t, p = _scipy_stats.ttest_rel(a, b)
        return float(t), float(p)
    d = a - b
    m, s = d.mean(), d.std(ddof=1)
    if s <= 0.0:
        return (0.0, 1.0) if m == 0.0 else (float("inf"), 0.0)
    t = m / (s / math.sqrt(len(d)))
    return float(t), _normal_p(t)


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test -> (statistic, p).  scipy preferred."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[d != 0.0]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    if HAVE_SCIPY:
        try:
            w, p = _scipy_stats.wilcoxon(d)
            return float(w), float(p)
        except ValueError:
            return float("nan"), float("nan")
    ranks = _rankdata(np.abs(d))
    w = float(np.sum(ranks[d > 0.0]))
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu) / sd if sd > 0 else 0.0
    return w, _normal_p(z)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Pooled Cohen's d (positive -> a greater than b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled <= 0.0:
        return 0.0 if a.mean() == b.mean() else float("nan")
    return float((a.mean() - b.mean()) / pooled)


def bootstrap_ci_mean_diff(a: Sequence[float], b: Sequence[float],
                           n_boot: int = BOOTSTRAP_N,
                           seed: int = BOOTSTRAP_SEED,
                           alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap 95% CI for the mean of paired differences (a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan"), float("nan")
    d = a - b
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(d), len(d))
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_mean(vals: Sequence[float], n_boot: int = BOOTSTRAP_N,
                      seed: int = BOOTSTRAP_SEED,
                      alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap 95% CI for the mean of one sample."""
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(v), len(v))
        means[i] = v[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ====================================================================== #
# Data loading (defensive)                                               #
# ====================================================================== #

def _f(rec: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_json(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] failed to read {path}: {e}")
        return None


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map any record dict (round-log row or memory record) to a canonical
    per-round dict with numeric fields coerced."""
    r = dict(rec)
    r["round"] = int(_num(_f(rec, *KEY_ROUND, default=0)))
    r["weight"] = _num(_f(rec, *KEY_WEIGHT, default=0.0))
    r["target"] = int(_num(_f(rec, *KEY_TARGET, default=-1)))
    r["applied_change"] = _num(_f(rec, *KEY_APPLIED, default=0.0))
    r["probe_delta"] = _num(_f(rec, *KEY_PROBE, default=0.0))
    r["outcome"] = str(_f(rec, "outcome", "result", default="neutral"))
    r["category"] = str(_f(rec, "category", default=r["outcome"]))
    r["injected"] = bool(_f(rec, "injected", default=False))
    r["correction_applied"] = bool(_f(rec, "correction_applied", default=False))
    r["correction_norm"] = _num(_f(rec, "correction_norm", default=0.0))
    r["theta_norm"] = _num(_f(rec, "theta_norm", default=float("nan")))
    r["memory_similarity"] = _f(rec, *KEY_MEM_SIM)
    r["future_similarity"] = _f(rec, *KEY_FUT_SIM)
    r["correction_similarity"] = _f(rec, *KEY_CORR_SIM)
    r["memory_age"] = _f(rec, *KEY_MEM_AGE)
    r["experience_id"] = _f(rec, *KEY_EXP_ID)
    r["source_experience_ids"] = _f(rec, *KEY_SOURCE, default=[])
    # direction vector (v0.5.2) if the runner stored it
    for dk in ("direction", "z_direction", "delta_theta", "correction_direction"):
        if dk in rec and rec[dk] is not None:
            r.setdefault("direction", rec[dk])
    return r


def round_records_from_round_log(round_log: Any) -> List[Dict[str, Any]]:
    if not isinstance(round_log, list) or not round_log:
        return []
    return [normalize_record(r) for r in round_log if isinstance(r, dict)]


def round_records_from_memory(memory: Any) -> List[Dict[str, Any]]:
    if not isinstance(memory, dict):
        return []
    recs = memory.get("records") or []
    return [normalize_record(r) for r in recs if isinstance(r, dict)]


def merge_records(round_log_recs: List[Dict[str, Any]],
                  memory_recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge round-log and memory records by round; round-log wins but
    memory fills fields the round log lacks (e.g. experience lineage)."""
    by_round: Dict[int, Dict[str, Any]] = {}
    for r in round_log_recs:
        by_round[r["round"]] = dict(r)
    for r in memory_recs:
        rid = r["round"]
        if rid in by_round:
            base = by_round[rid]
            for k, v in r.items():
                if k not in base or base[k] in (None, ""):
                    base[k] = v
        else:
            by_round[rid] = dict(r)
    return [by_round[k] for k in sorted(by_round)]


def load_seed_dir(seed_dir: str) -> Dict[str, Any]:
    """Load everything stored for one (condition, seed) directory."""
    data: Dict[str, Any] = {"result": None, "rounds": [], "memory": None,
                            "lineage": None, "future": None, "absorption": None}
    data["result"] = load_json(os.path.join(seed_dir, "result.json"))
    rl = load_json(os.path.join(seed_dir, "round_log.json"))
    mem = load_json(os.path.join(seed_dir, "memory.json"))
    rl_recs = round_records_from_round_log(rl) if rl is not None else []
    mem_recs = round_records_from_memory(mem) if mem is not None else []
    data["rounds"] = merge_records(rl_recs, mem_recs)
    data["memory"] = mem
    # v0.5.2 optional artifacts
    for key, names in (
        ("lineage", ("lineage.json", "lineages.json", "experiences.json",
                     "experience_lineage.json")),
        ("future", ("future_behavior.json", "future.json")),
        ("absorption", ("absorption.json", "absorption_result.json")),
    ):
        for name in names:
            blob = load_json(os.path.join(seed_dir, name))
            if blob is not None:
                data[key] = blob
                break
    return data


def parse_seed_dirname(dname: str) -> Optional[Tuple[str, int]]:
    parts = dname.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    cond, seed_str = parts
    if not seed_str.isdigit():
        return None
    return cond, int(seed_str)


def load_all(input_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {condition: {seed: seed_data}} for every available run."""
    results_dir = os.path.join(input_dir, "results")
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not os.path.isdir(results_dir):
        print(f"  [warn] results dir not found: {results_dir}")
        return out
    for dname in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, dname)
        if not os.path.isdir(full):
            continue
        parsed = parse_seed_dirname(dname)
        if parsed is None:
            continue
        cond, seed = parsed
        out.setdefault(cond, {})[seed] = load_seed_dir(full)
    # attach per-condition summary files if present
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_summary.json"):
            cond = fname[: -len("_summary.json")]
            if cond in out:
                blob = load_json(os.path.join(results_dir, fname))
                if blob is not None:
                    for sd in out[cond].values():
                        sd["summary"] = blob
    return out


def conditions_present(all_data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    order = [c for c in EXPERIMENT_CONDITIONS if c in all_data]
    extras = sorted(c for c in all_data if c not in EXPERIMENT_CONDITIONS)
    return order + extras


# ====================================================================== #
# Metric helpers on a set of round records                              #
# ====================================================================== #

def is_failure(rec: Dict[str, Any]) -> bool:
    return rec["category"] == "failure" or rec["outcome"] == "failure"


def is_success(rec: Dict[str, Any]) -> bool:
    return rec["category"] == "success" or rec["outcome"] in SUCCESS_OUTCOMES


def is_recovery(rec: Dict[str, Any]) -> bool:
    if rec["category"] == "recovery":
        return True
    # proxy: correction applied and round improved
    return bool(rec["correction_applied"]) and rec["probe_delta"] > 0


def srr_in(recs: Sequence[Dict[str, Any]]) -> float:
    recs = list(recs)
    n_fail = sum(1 for r in recs if is_failure(r))
    if n_fail == 0:
        return 0.0
    n_rec = sum(1 for r in recs if is_recovery(r))
    return n_rec / n_fail


def rfr_in(recs: Sequence[Dict[str, Any]], window: Optional[int] = None) -> float:
    """Repeat-failure rate: fraction of failures (after the first) whose
    target_group repeats a previous failure within ``window`` rounds."""
    recs = sorted((r for r in recs if is_failure(r)), key=lambda r: r["round"])
    if len(recs) < 2:
        return 0.0
    repeats = 0
    for i in range(1, len(recs)):
        prev, cur = recs[i - 1], recs[i]
        if window is None or (cur["round"] - prev["round"]) <= window:
            if prev["target"] >= 0 and prev["target"] == cur["target"]:
                repeats += 1
    return repeats / (len(recs) - 1)


def mean_weight_by(recs: Sequence[Dict[str, Any]], kind: str) -> float:
    if kind == "success":
        sel = [r["weight"] for r in recs if is_success(r)]
    elif kind == "failure":
        sel = [r["weight"] for r in recs if is_failure(r)]
    else:
        sel = [r["weight"] for r in recs]
    return float(np.mean(sel)) if sel else 0.0


def cumulative_srr(recs: Sequence[Dict[str, Any]]) -> List[float]:
    """Cumulative SRR curve: recoveries / failures up to each round."""
    recs = sorted(recs, key=lambda r: r["round"])
    curve, fails, recs_n = [], 0, 0
    for r in recs:
        if is_failure(r):
            fails += 1
        if is_recovery(r):
            recs_n += 1
        curve.append(recs_n / fails if fails > 0 else 0.0)
    return curve


def running_window_srr(recs: Sequence[Dict[str, Any]],
                       window: int = RUNNING_WINDOW) -> List[float]:
    recs = sorted(recs, key=lambda r: r["round"])
    curve = []
    for i in range(len(recs)):
        lo = max(0, i - window + 1)
        curve.append(srr_in(recs[lo:i + 1]))
    return curve


def running_window_rfr(recs: Sequence[Dict[str, Any]],
                       window: int = RFR_WINDOW) -> List[float]:
    recs = sorted(recs, key=lambda r: r["round"])
    curve = []
    for i in range(len(recs)):
        lo = max(0, i - window + 1)
        curve.append(rfr_in(recs[lo:i + 1], window=window))
    return curve


def running_failure_rate(recs: Sequence[Dict[str, Any]],
                         window: int = RUNNING_WINDOW) -> List[float]:
    recs = sorted(recs, key=lambda r: r["round"])
    curve = []
    for i in range(len(recs)):
        lo = max(0, i - window + 1)
        chunk = recs[lo:i + 1]
        curve.append(sum(1 for r in chunk if is_failure(r)) / len(chunk))
    return curve


def running_mean_of(recs: Sequence[Dict[str, Any]], picker,
                    window: int = RUNNING_WINDOW) -> List[float]:
    """Running mean of a per-record numeric value (picker: rec -> float or None)."""
    recs = sorted(recs, key=lambda r: r["round"])
    curve = []
    for i in range(len(recs)):
        lo = max(0, i - window + 1)
        vals = [v for r in recs[lo:i + 1]
                if (v := picker(r)) is not None]
        curve.append(float(np.mean(vals)) if vals else float("nan"))
    return curve


def cumulative_sum(recs: Sequence[Dict[str, Any]], picker) -> List[float]:
    recs = sorted(recs, key=lambda r: r["round"])
    curve, acc = [], 0.0
    for r in recs:
        v = picker(r)
        if v is None:
            v = 0.0
        acc += v
        curve.append(acc)
    return curve


def memory_age_curve(recs: Sequence[Dict[str, Any]]) -> List[float]:
    """Mean age (current round - record round) of all memory records."""
    recs = sorted(recs, key=lambda r: r["round"])
    curve = []
    for i in range(len(recs)):
        cur_round = recs[i]["round"]
        ages = [cur_round - recs[j]["round"] for j in range(i + 1)]
        curve.append(float(np.mean(ages)) if ages else 0.0)
    return curve


# ====================================================================== #
# Per-seed metric extraction from result.json (fallback: computed)       #
# ====================================================================== #

def extract_seed_metrics(seed_data: Dict[str, Any]) -> Dict[str, float]:
    """Best-effort per-seed scalar metrics used by the stats section."""
    res = seed_data.get("result") or {}
    recs = seed_data.get("rounds") or []
    m: Dict[str, float] = {}
    # --- direct from result.json ---
    for key in ("SRR", "RFR_similar", "RFR_target", "RFR_exact",
                "w_after_success", "w_after_failure", "weight_adaptation",
                "failure_rate", "recovery_rate", "correction_rate",
                "net_drift", "gross_drift", "CAR", "RE",
                "natural_failure_rate", "EAR", "absorption_rate",
                "successful_reuse_rate", "repeat_failure_rate"):
        if key in res:
            m[key] = _num(res[key])
    # --- fallback: compute from round records ---
    if "SRR" not in m and recs:
        m["SRR"] = srr_in(recs)
    if "RFR_similar" not in m and recs:
        m["RFR_similar"] = rfr_in(recs, window=RFR_WINDOW)
    if "w_after_success" not in m and recs:
        m["w_after_success"] = mean_weight_by(recs, "success")
    if "w_after_failure" not in m and recs:
        m["w_after_failure"] = mean_weight_by(recs, "failure")
    if "weight_adaptation" not in m:
        m["weight_adaptation"] = m.get("w_after_success", 0.0) - m.get(
            "w_after_failure", 0.0)
    if "failure_rate" not in m and recs:
        m["failure_rate"] = sum(1 for r in recs if is_failure(r)) / max(
            len(recs), 1)
    return m


# ====================================================================== #
# Figures (12 long-term curves)                                          #
# ====================================================================== #

def _import_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _series_for(all_data, cond, picker):
    """Return (rounds, matrix[seed, round]) for a numeric picker, or None."""
    seeds = sorted(all_data[cond])
    runs = []
    for sd in seeds:
        recs = sorted(all_data[cond][sd].get("rounds") or [],
                      key=lambda r: r["round"])
        if not recs:
            continue
        curve = picker(recs)
        if not curve:
            continue
        runs.append(([r["round"] for r in recs], np.asarray(curve, dtype=float)))
    if not runs:
        return None
    # align to a common round axis (max length)
    max_len = max(len(rs) for rs, _ in runs)
    mat = np.full((len(runs), max_len), np.nan)
    for i, (rs, cv) in enumerate(runs):
        for j, (rnd, v) in enumerate(zip(rs, cv)):
            if j < max_len:
                mat[i, j] = v
    rounds = list(range(1, max_len + 1))
    return rounds, mat


def _plot_mean_std(plt, rounds, mat, ax, label, color=None, marker=None):
    if mat.shape[0] == 0:
        return
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    finite = np.isfinite(mean)
    xs = np.asarray(rounds, dtype=float)[finite]
    ys = mean[finite]
    ax.plot(xs, ys, label=label, color=color, marker=marker, markersize=2,
            linewidth=1.5, alpha=0.9)
    if mat.shape[0] >= 2:
        std_f = std[finite]
        ax.fill_between(xs, ys - std_f, ys + std_f, color=color, alpha=0.15)


def _single_point_curve(plt, all_data, cond, ax, key, color):
    """Plot per-seed final scalars (from result.json) as a horizontal line
    when no round data exists."""
    seeds = sorted(all_data[cond])
    vals = []
    for sd in seeds:
        m = extract_seed_metrics(all_data[cond][sd])
        if key in m:
            vals.append(m[key])
    if vals:
        ax.plot([0, len(seeds)], [np.mean(vals), np.mean(vals)],
                color=color, linestyle=":", linewidth=1.2,
                label=f"{cond} (final mean, no round data)")


def make_long_term_curves(all_data, output_dir: str) -> List[str]:
    """Produce the 12 long-term curves (§28). Returns figure filenames."""
    try:
        plt = _import_pyplot()
    except ImportError:
        print("  [warn] matplotlib not available; skipping figures")
        return []
    os.makedirs(output_dir, exist_ok=True)
    conds = conditions_present(all_data)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conds), 1)))

    # curve definitions: (title, ylabel, fname, picker, final_scalar_key)
    def _picker_net_drift(recs):
        # per-round explicit net_drift if present, else cumulative applied change
        explicit = [r.get("net_drift") for r in recs
                    if r.get("net_drift") is not None]
        if len(explicit) == len(recs):
            return list(explicit)
        return cumulative_sum(recs, lambda r: r.get("applied_change"))

    def _picker_probe_drift(recs):
        explicit = [r.get("probe_drift") for r in recs
                    if r.get("probe_drift") is not None]
        if len(explicit) == len(recs):
            return list(explicit)
        return cumulative_sum(recs, lambda r: r.get("probe_delta"))

    def _picker_memory_sim(recs):
        return running_mean_of(recs, lambda r: _opt(r, "memory_similarity"))

    def _picker_future_sim(recs):
        return running_mean_of(recs, lambda r: _opt(r, "future_similarity"))

    def _picker_corr_sim(recs):
        # explicit direction-similarity if stored; else correction norm proxy
        explicit = running_mean_of(recs, lambda r: _opt(r, "correction_similarity"))
        if any(np.isfinite(v) for v in explicit):
            return explicit
        return running_mean_of(recs, lambda r: (
            r["correction_norm"] if r["correction_applied"] else None))

    def _picker_mem_age(recs):
        explicit = running_mean_of(recs, lambda r: _opt(r, "memory_age"))
        if any(np.isfinite(v) for v in explicit):
            return explicit
        return memory_age_curve(recs)

    def _picker_w(recs):
        return running_mean_of(recs, lambda r: r["weight"])

    def _picker_w_kind(kind):
        return lambda recs: running_mean_of(
            recs, lambda r: r["weight"] if (
                (kind == "success" and is_success(r)) or
                (kind == "failure" and is_failure(r))) else None)

    curve_defs = [
        ("1. Net Drift vs Round", "Net drift (cumulative ||Δθ||)",
         "c01_net_drift.png", _picker_net_drift, "net_drift"),
        ("2. Probe Drift vs Round", "Probe drift (cumulative Δscore)",
         "c02_probe_drift.png", _picker_probe_drift, "probe_drift_final"),
        ("3. Modification Magnitude vs Round", "Weight / magnitude",
         "c03_modification_magnitude.png", _picker_w, "w_after_success"),
        ("4. Failure Rate vs Round (running)", "Running failure rate",
         "c04_failure_rate.png", lambda recs: running_failure_rate(recs),
         "failure_rate"),
        ("5. SRR vs Round (cumulative)", "Cumulative SRR",
         "c05_srr.png", cumulative_srr, "SRR"),
        ("6. RFR_similar vs Round (running)", "Running RFR_similar",
         "c06_rfr_similar.png", lambda recs: running_window_rfr(recs),
         "RFR_similar"),
        ("7. Memory Retrieval Similarity vs Round", "Mean retrieval similarity",
         "c07_memory_retrieval_similarity.png", _picker_memory_sim, None),
        ("8. w_success vs Round", "Weight on success rounds",
         "c08_w_success.png", _picker_w_kind("success"), "w_after_success"),
        ("9. w_failure vs Round", "Weight on failure rounds",
         "c09_w_failure.png", _picker_w_kind("failure"), "w_after_failure"),
        ("10. Future Modification Similarity vs Round",
         "Mean similarity to history", "c10_future_mod_similarity.png",
         _picker_future_sim, None),
        ("11. Correction Direction Similarity vs Round",
         "Correction direction similarity (or ||corr|| proxy)",
         "c11_correction_similarity.png", _picker_corr_sim, None),
        ("12. Memory Age vs Round", "Mean memory age (rounds)",
         "c12_memory_age.png", _picker_mem_age, None),
    ]

    made = []
    for idx, (title, ylabel, fname, picker, scalar_key) in enumerate(curve_defs):
        fig, ax = plt.subplots(figsize=(11, 6))
        any_plotted = False
        for ci, cond in enumerate(conds):
            series = _series_for(all_data, cond, picker)
            if series is None:
                if scalar_key:
                    _single_point_curve(plt, all_data, cond, ax,
                                        scalar_key, colors[ci])
                    any_plotted = True
                continue
            rounds, mat = series
            _plot_mean_std(plt, rounds, mat, ax, cond, colors[ci])
            any_plotted = True
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"v0.5.2 Long-Term Curve {title} (mean ± std over seeds)")
        if any_plotted:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(fname)
        print(f"    wrote {fname}")
    return made


def _opt(rec: Dict[str, Any], key: str) -> Optional[float]:
    v = rec.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ====================================================================== #
# 4-phase analysis (§25)                                                 #
# ====================================================================== #

def phase_analysis(all_data, phases, output_dir: str) -> Dict[str, Any]:
    """Mean SRR / RFR / w_success / w_failure per phase and condition."""
    os.makedirs(output_dir, exist_ok=True)
    conds = conditions_present(all_data)
    rows: List[Dict[str, Any]] = []
    for cond in conds:
        for pi, (lo, hi) in enumerate(phases):
            per_seed: Dict[str, List[float]] = {
                "SRR": [], "RFR": [], "w_success": [], "w_failure": []}
            n_seed = 0
            for sd, sdata in sorted(all_data[cond].items()):
                recs = [r for r in (sdata.get("rounds") or [])
                        if lo <= r["round"] <= hi]
                if not recs:
                    continue
                n_seed += 1
                per_seed["SRR"].append(srr_in(recs))
                per_seed["RFR"].append(rfr_in(recs, window=RFR_WINDOW))
                per_seed["w_success"].append(mean_weight_by(recs, "success"))
                per_seed["w_failure"].append(mean_weight_by(recs, "failure"))
            for metric in ("SRR", "RFR", "w_success", "w_failure"):
                vals = per_seed[metric]
                rows.append({
                    "condition": cond, "phase": pi + 1,
                    "phase_name": PHASE_NAMES[pi] if pi < len(PHASE_NAMES)
                                  else f"R{lo}-{hi}",
                    "rounds": f"R{lo}-{hi}",
                    "metric": metric, "mean": float(np.mean(vals)) if vals
                    else float("nan"), "std": float(np.std(vals)) if vals
                    else float("nan"), "n_seeds": n_seed,
                })
    # ---- CSV ----
    csv_path = os.path.join(output_dir, "phase_analysis.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                           else ["condition", "phase", "phase_name", "rounds",
                                 "metric", "mean", "std", "n_seeds"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    # ---- markdown ----
    md_lines = ["# v0.5.2 4-Phase Analysis (§25)\n"]
    if rows:
        md_lines.append("| Condition | Phase | Rounds | Metric | Mean | Std | N seeds |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            md_lines.append(
                f"| {row['condition']} | {row['phase']} | {row['rounds']} "
                f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} "
                f"| {row['n_seeds']} |")
    md_path = os.path.join(output_dir, "phase_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"    wrote {os.path.basename(csv_path)}, {os.path.basename(md_path)}")

    # ---- figure: comparison table + grouped bars ----
    try:
        plt = _import_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        # heatmap: mean SRR by condition x phase
        metrics_hm = ["SRR", "RFR", "w_success", "w_failure"]
        n_phases = len(phases)
        hm = np.full((len(conds), n_phases), np.nan)
        for ci, cond in enumerate(conds):
            for pi in range(n_phases):
                vals = [r["mean"] for r in rows
                        if r["condition"] == cond and r["phase"] == pi + 1
                        and r["metric"] == "SRR"]
                if vals:
                    hm[ci, pi] = vals[0]
        im = axes[0].imshow(hm, cmap="viridis", aspect="auto")
        axes[0].set_xticks(range(n_phases))
        axes[0].set_xticklabels([f"R{lo}-{hi}" for lo, hi in phases],
                                rotation=30)
        axes[0].set_yticks(range(len(conds)))
        axes[0].set_yticklabels(conds)
        axes[0].set_title("Mean SRR by Phase")
        axes[0].grid(False)
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        # grouped bars: w_success vs w_failure per phase for Full
        if FULL_CONDITION in all_data:
            x = np.arange(n_phases)
            ws = [next((r["mean"] for r in rows
                        if r["condition"] == FULL_CONDITION and r["phase"] == pi + 1
                        and r["metric"] == "w_success"), np.nan)
                  for pi in range(n_phases)]
            wf = [next((r["mean"] for r in rows
                        if r["condition"] == FULL_CONDITION and r["phase"] == pi + 1
                        and r["metric"] == "w_failure"), np.nan)
                  for pi in range(n_phases)]
            axes[1].bar(x - 0.2, ws, 0.4, label="w_success", color="#2ca02c")
            axes[1].bar(x + 0.2, wf, 0.4, label="w_failure", color="#d62728")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"R{lo}-{hi}" for lo, hi in phases],
                                    rotation=30)
            axes[1].set_ylabel("Weight")
            axes[1].set_title(f"{FULL_CONDITION}: outcome-conditioned weight")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis="y")
        fig.suptitle("v0.5.2 Phase Analysis (§25)")
        fig.tight_layout()
        fname = "phase_comparison.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"    wrote {fname}")
    except ImportError:
        pass
    return {"rows": rows, "phases": [f"R{lo}-{hi}" for lo, hi in phases]}


# ====================================================================== #
# Statistical tests (§43) + per-seed breakdown (§44)                     #
# ====================================================================== #

def _paired_stats(cond_a_vals, cond_b_vals, metric):
    """Compare two aligned per-seed value lists."""
    if not cond_a_vals or not cond_b_vals:
        return None
    # align by seed index (they are already parallel lists if we keep order)
    a = np.asarray(cond_a_vals, dtype=float)
    b = np.asarray(cond_b_vals, dtype=float)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    if len(a) < 2:
        return None
    t, p_t = ttest_paired(a, b)
    w, p_w = wilcoxon_paired(a, b)
    d = cohens_d(a, b)
    lo, hi = bootstrap_ci_mean_diff(a, b)
    return {
        "metric": metric,
        "n_seeds": int(len(a)),
        "mean_a": float(a.mean()), "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()), "std_b": float(b.std(ddof=1)),
        "mean_diff": float((a - b).mean()),
        "t": t, "p_ttest": p_t,
        "wilcoxon_stat": w, "p_wilcoxon": p_w,
        "cohens_d": d,
        "ci95_diff": [lo, hi],
        "significant": bool(p_t < 0.05 and p_w < 0.05) if (
            not math.isnan(p_t) and not math.isnan(p_w)) else False,
    }


def statistical_tests(all_data, output_dir: str) -> Dict[str, Any]:
    """Paired tests: Full vs every other present condition, per metric."""
    os.makedirs(output_dir, exist_ok=True)
    conds = conditions_present(all_data)
    metrics = ["SRR", "RFR_similar", "w_after_success", "w_after_failure",
               "failure_rate", "net_drift"]
    full_seeds = sorted(all_data.get(FULL_CONDITION, {}))
    results: Dict[str, Any] = {
        "method": "paired t-test, Wilcoxon signed-rank, Cohen's d, "
                  "bootstrap 95% CI of mean difference",
        "scipy_available": HAVE_SCIPY,
        "comparisons": [], "per_seed": {},
    }

    def _per_seed_vals(cond, metric):
        vals, seeds = [], []
        for sd in sorted(all_data[cond]):
            m = extract_seed_metrics(all_data[cond][sd])
            if metric in m:
                vals.append(m[metric])
                seeds.append(sd)
        return seeds, vals

    # per-seed breakdown (§44) — every condition, every metric
    per_seed_rows: List[Dict[str, Any]] = []
    for cond in conds:
        for metric in metrics:
            seeds, vals = _per_seed_vals(cond, metric)
            if not vals:
                continue
            results["per_seed"].setdefault(cond, {})[metric] = {
                "seeds": seeds, "values": vals,
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            }
            for sd, v in zip(seeds, vals):
                per_seed_rows.append({"condition": cond, "seed": sd,
                                      "metric": metric, "value": v})

    # comparisons: Full vs each baseline (aligned on shared seeds)
    for cond in conds:
        if cond == FULL_CONDITION or cond not in all_data:
            continue
        if not all_data[cond]:
            print(f"  [skip] {cond}: no seed data")
            continue
        for metric in metrics:
            seeds_a, vals_a = _per_seed_vals(FULL_CONDITION, metric)
            seeds_b, vals_b = _per_seed_vals(cond, metric)
            shared = sorted(set(seeds_a) & set(seeds_b))
            if len(shared) < 2:
                continue
            va = [dict(zip(seeds_a, vals_a))[s] for s in shared]
            vb = [dict(zip(seeds_b, vals_b))[s] for s in shared]
            row = _paired_stats(va, vb, metric)
            if row:
                row.update({"condition_a": FULL_CONDITION,
                            "condition_b": cond, "seeds": shared})
                results["comparisons"].append(row)

    # ---- write CSV: per-seed breakdown (§44) ----
    csv_path = os.path.join(output_dir, "per_seed_breakdown.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["condition", "seed", "metric",
                                          "value"])
        w.writeheader()
        for row in per_seed_rows:
            w.writerow(row)
    # markdown per-seed table
    md = ["# v0.5.2 Per-Seed Breakdown (§44)\n",
          "Raw per-seed values for every available condition/metric.\n"]
    for cond in conds:
        if cond not in results["per_seed"]:
            continue
        md.append(f"\n## {cond}\n")
        md.append("| Seed | " + " | ".join(
            [f"{m}" for m in metrics if m in results["per_seed"][cond]]) + " |")
        md.append("|---|" + "---|" * sum(
            1 for m in metrics if m in results["per_seed"][cond]))
        seeds_all = sorted(results["per_seed"][cond].get("SRR", {}).get("seeds",
                              results["per_seed"][cond].get(
                                  list(results["per_seed"][cond])[0],
                                  {}).get("seeds", [])))
        for sd in seeds_all:
            row = [f"{sd}"]
            for metric in metrics:
                blob = results["per_seed"][cond].get(metric)
                if blob and sd in blob["seeds"]:
                    v = dict(zip(blob["seeds"], blob["values"]))[sd]
                    row.append(f"{v:.4f}")
            md.append("| " + " | ".join(row) + " |")
    md_path = os.path.join(output_dir, "per_seed_breakdown.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"    wrote {os.path.basename(csv_path)}, "
          f"{os.path.basename(md_path)}")

    # ---- markdown report ----
    rep = ["# v0.5.2 Statistical Analysis Report (§43)\n",
           f"- Paired tests: {FULL_CONDITION} vs each baseline (by seed).",
           f"- Tests: paired t-test, Wilcoxon signed-rank, Cohen's d, "
           f"bootstrap 95% CI of mean diff (n={BOOTSTRAP_N}).",
           f"- scipy available: {HAVE_SCIPY}\n"]
    if results["comparisons"]:
        rep.append("| Comparison | Metric | Full μ±σ | Base μ±σ | Δ | t (p) | "
                   "W (p) | d | 95% CI Δ | sig |")
        rep.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in results["comparisons"]:
            rep.append(
                f"| {r['condition_a']} vs {r['condition_b']} | {r['metric']} "
                f"| {r['mean_a']:.4f}±{r['std_a']:.4f} "
                f"| {r['mean_b']:.4f}±{r['std_b']:.4f} "
                f"| {r['mean_diff']:+.4f} "
                f"| {r['t']:.3f} ({r['p_ttest']:.4f}) "
                f"| {r['wilcoxon_stat']:.3f} ({r['p_wilcoxon']:.4f}) "
                f"| {r['cohens_d']:+.3f} "
                f"| [{r['ci95_diff'][0]:.4f}, {r['ci95_diff'][1]:.4f}] "
                f"| {'**yes**' if r['significant'] else 'no'} |")
    else:
        rep.append("_No paired comparisons possible: "
                   "fewer than 2 shared seeds with Full._")
    rep_path = os.path.join(output_dir, "statistical_report.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"    wrote {os.path.basename(rep_path)}")

    json_path = os.path.join(output_dir, "statistical_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"    wrote {os.path.basename(json_path)}")

    # ---- figure: effect sizes ----
    try:
        plt = _import_pyplot()
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"{r['condition_b']}·{r['metric']}"
                  for r in results["comparisons"]]
        ds = [r["cohens_d"] for r in results["comparisons"]]
        sigs = [r["significant"] for r in results["comparisons"]]
        colors = ["#d62728" if s else "#7f7f7f" for s in sigs]
        if labels:
            ax.bar(range(len(labels)), ds, color=colors)
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_ylabel("Cohen's d (Full − baseline)")
            ax.set_title("v0.5.2 Effect sizes (§43); red = significant "
                         "(t & W both p<0.05)")
            ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fname = "effect_sizes.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"    wrote {fname}")
    except ImportError:
        pass

    return results


# ====================================================================== #
# Experience absorption curves                                           #
# ====================================================================== #

def compute_ear(rfr_baseline: float, rfr_future: float) -> float:
    """EAR = 1 - RFR_future / RFR_baseline, baseline-0 safe."""
    if rfr_baseline <= 0.0:
        if rfr_future <= 0.0:
            return 0.0
        return -1.0
    return float(np.clip(1.0 - rfr_future / rfr_baseline, -1.0, 1.0))


def ear_curve(recs, window: int = EAR_WINDOW) -> List[float]:
    """EAR(t): baseline RFR on the first window, future RFR on window ending t."""
    recs = sorted(recs, key=lambda r: r["round"])
    if len(recs) < 2 * window:
        return []
    baseline = rfr_in(recs[:window], window=window)
    curve = []
    for i in range(window, len(recs) + 1):
        future = rfr_in(recs[i - window:i], window=window)
        curve.append(compute_ear(baseline, future))
    return curve


def build_lineage_from_records(recs: Sequence[Dict[str, Any]]):
    """Synthesize failure->future-success lineage from round records when the
    runner did not store lineage.json.

    A failure at round f "links" to every later record whose target_group
    matches the failure's target (a proxy for 'used that experience').
    """
    recs = sorted(recs, key=lambda r: r["round"])
    failures = [r for r in recs if is_failure(r)]
    lineages = []
    for f in failures:
        linked = [r for r in recs
                  if r["round"] > f["round"] and r["target"] == f["target"]]
        future_rounds = [r["round"] for r in linked]
        future_outcomes = [r["outcome"] for r in linked]
        n_successes = sum(1 for r in linked if is_success(r))
        lineages.append({
            "failure_round": f["round"],
            "failure_target": f["target"],
            "n_uses": len(linked),
            "n_successes": n_successes,
            "efficacy": (n_successes / len(linked)) if linked else 0.0,
            "future_rounds": future_rounds,
            "future_outcomes": future_outcomes,
        })
    return lineages


def load_lineage(seed_data: Dict[str, Any]):
    """Prefer stored lineage.json; else synthesize from records."""
    lin = seed_data.get("lineage")
    if lin is not None:
        return lin
    recs = seed_data.get("rounds") or []
    if recs:
        return {"synthesized": True,
                "lineages": build_lineage_from_records(recs)}
    return None


def absorption_analysis(all_data, output_dir: str) -> Dict[str, Any]:
    """EAR over time, failure lineage visualization, success reuse rate."""
    os.makedirs(output_dir, exist_ok=True)
    conds = conditions_present(all_data)
    summary: Dict[str, Any] = {}

    # ---------------- EAR over time ----------------
    ear_series = {}
    for cond in conds:
        curves = []
        for sd, sdata in sorted(all_data[cond].items()):
            recs = sdata.get("rounds") or []
            if len(recs) < 2 * EAR_WINDOW:
                continue
            curves.append(ear_curve(recs))
        if curves:
            n = min(len(c) for c in curves)
            ear_series[cond] = {
                "rounds": list(range(EAR_WINDOW + 1, EAR_WINDOW + n + 1)),
                "curves": [c[:n] for c in curves],
                "mean": [float(np.mean([c[i] for c in curves]))
                         for i in range(n)],
                "std": [float(np.std([c[i] for c in curves]))
                        for i in range(n)],
            }
    try:
        plt = _import_pyplot()
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(conds), 1)))
        any_plot = False
        for ci, cond in enumerate(conds):
            es = ear_series.get(cond)
            if not es:
                continue
            ax.plot(es["rounds"], es["mean"], label=cond, color=colors[ci],
                    linewidth=1.5)
            ax.fill_between(es["rounds"],
                            np.asarray(es["mean"]) - np.asarray(es["std"]),
                            np.asarray(es["mean"]) + np.asarray(es["std"]),
                            color=colors[ci], alpha=0.12)
            any_plot = True
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Round")
        ax.set_ylabel("EAR = 1 − RFR_future / RFR_baseline")
        ax.set_title("v0.5.2 Experience Absorption Rate over time "
                     "(EAR > 0 ⇒ failures absorbed)")
        if any_plot:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "absorption_ear.png"), dpi=150)
        plt.close(fig)
        print("    wrote absorption_ear.png")
    except ImportError:
        pass

    # ---------------- lineage ----------------
    lin_data: Dict[str, List[Any]] = {}
    for cond in conds:
        for sd, sdata in sorted(all_data[cond].items()):
            lin = load_lineage(sdata)
            if lin is None:
                continue
            items = lin.get("lineages") or lin.get("experiences") or []
            # ExperienceTracker.to_dict stores 'lineages' as list of dicts
            if isinstance(items, list):
                lin_data.setdefault(cond, []).extend(items)
    try:
        plt = _import_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(conds), 1)))
        any_lin = False
        for ci, cond in enumerate(conds):
            items = lin_data.get(cond) or []
            if not items:
                continue
            fr = [i.get("failure_round", i.get("failure_round_id", 0))
                  for i in items]
            nuse = [i.get("n_uses", i.get("n_uses_count", 0)) for i in items]
            eff = [i.get("efficacy", 0.0) for i in items]
            axes[0].scatter(fr, nuse, s=30, alpha=0.6, label=cond,
                            color=colors[ci])
            axes[1].scatter(fr, eff, s=30, alpha=0.6, label=cond,
                            color=colors[ci])
            any_lin = True
        axes[0].set_xlabel("Failure round")
        axes[0].set_ylabel("Linked future modifications")
        axes[0].set_title("Failure lineage: usage count")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Failure round")
        axes[1].set_ylabel("Efficacy (frac. of links that succeeded)")
        axes[1].set_title("Failure lineage: efficacy")
        axes[1].grid(True, alpha=0.3)
        if any_lin:
            axes[0].legend(fontsize=7)
            axes[1].legend(fontsize=7)
        fig.suptitle("v0.5.2 Failure lineage tracking")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "lineage_tracking.png"), dpi=150)
        plt.close(fig)
        print("    wrote lineage_tracking.png")
    except ImportError:
        pass

    # ---------------- success reuse rate over time ----------------
    reuse_series = {}
    for cond in conds:
        curves = []
        for sd, sdata in sorted(all_data[cond].items()):
            recs = sdata.get("rounds") or []
            if len(recs) < RUNNING_WINDOW:
                continue
            curves.append(_success_reuse_curve(recs))
        if curves:
            n = min(len(c) for c in curves)
            reuse_series[cond] = {
                "rounds": list(range(1, n + 1)),
                "mean": [float(np.mean([c[i] for c in curves]))
                         for i in range(n)],
                "std": [float(np.std([c[i] for c in curves]))
                        for i in range(n)],
            }
    try:
        plt = _import_pyplot()
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(conds), 1)))
        any_reuse = False
        for ci, cond in enumerate(conds):
            rs = reuse_series.get(cond)
            if not rs:
                continue
            ax.plot(rs["rounds"], rs["mean"], label=cond, color=colors[ci],
                    linewidth=1.5)
            ax.fill_between(rs["rounds"],
                            np.asarray(rs["mean"]) - np.asarray(rs["std"]),
                            np.asarray(rs["mean"]) + np.asarray(rs["std"]),
                            color=colors[ci], alpha=0.12)
            any_reuse = True
        ax.set_xlabel("Round")
        ax.set_ylabel("Success reuse rate (running)")
        ax.set_title("v0.5.2 Success reuse over time: P(SUCCESS | similar to "
                     "past success)")
        if any_reuse:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "success_reuse.png"), dpi=150)
        plt.close(fig)
        print("    wrote success_reuse.png")
    except ImportError:
        pass

    # ---------------- summary ----------------
    for cond in conds:
        ear_mean, ear_std = float("nan"), float("nan")
        if cond in ear_series and ear_series[cond]["mean"]:
            ear_mean = float(np.mean(ear_series[cond]["mean"]))
            ear_std = float(np.std(ear_series[cond]["mean"]))
        reuse_mean = float("nan")
        if cond in reuse_series and reuse_series[cond]["mean"]:
            reuse_mean = float(np.mean(reuse_series[cond]["mean"]))
        # overall EAR from phase-1 vs phase-4 RFR
        ov_ear, rfr_base, rfr_fut = float("nan"), float("nan"), float("nan")
        phases_ok = True
        seed_ears = []
        for sd, sdata in sorted(all_data[cond].items()):
            recs = sdata.get("rounds") or []
            if len(recs) < 300:
                phases_ok = False
                break
            base = rfr_in([r for r in recs if r["round"] <= 50],
                          window=RFR_WINDOW)
            fut = rfr_in([r for r in recs if r["round"] >= 301],
                         window=RFR_WINDOW)
            seed_ears.append(compute_ear(base, fut))
        if phases_ok and seed_ears:
            ov_ear = float(np.mean(seed_ears))
            rfr_base = float(np.mean([
                rfr_in([r for r in (all_data[cond][sd].get("rounds") or [])
                        if r["round"] <= 50], window=RFR_WINDOW)
                for sd in sorted(all_data[cond]) if all_data[cond][sd].get(
                    "rounds")]))
            rfr_fut = float(np.mean([
                rfr_in([r for r in (all_data[cond][sd].get("rounds") or [])
                        if r["round"] >= 301], window=RFR_WINDOW)
                for sd in sorted(all_data[cond]) if all_data[cond][sd].get(
                    "rounds")]))
        summary[cond] = {
            "ear_mean": ear_mean, "ear_std": ear_std,
            "ear_overall_phase14": ov_ear,
            "rfr_baseline_phase1": rfr_base,
            "rfr_future_phase4": rfr_fut,
            "success_reuse_mean": reuse_mean,
            "n_lineages": len(lin_data.get(cond) or []),
            "absorbed": bool(ov_ear > 0.0),
        }
    out_path = os.path.join(output_dir, "absorption_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    wrote {os.path.basename(out_path)}")
    return summary


def _success_reuse_curve(recs, window: int = RUNNING_WINDOW) -> List[float]:
    """Running fraction of success rounds whose target matches a previous
    success (target-based success-reuse proxy; real similarity when the
    future_behavior module data is present)."""
    recs = sorted(recs, key=lambda r: r["round"])
    curve = []
    past_success_targets: set = set()
    for i, r in enumerate(recs):
        if i >= window and is_success(recs[i - window]) and (
                recs[i - window]["target"] >= 0):
            # slide window: rebuild past-success set for window
            pass
        if is_success(r):
            reused = r["target"] >= 0 and r["target"] in past_success_targets
            curve.append(1.0 if reused else 0.0)
            past_success_targets.add(r["target"])
        else:
            curve.append(0.0)
    # running mean of the reuse indicator
    out = []
    for i in range(len(curve)):
        lo = max(0, i - window + 1)
        chunk = curve[lo:i + 1]
        out.append(float(np.mean(chunk)) if chunk else 0.0)
    return out


# ====================================================================== #
# Main                                                                   #
# ====================================================================== #

def main():
    ap = argparse.ArgumentParser(description="v0.5.2 analysis")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=None,
                    help="figures output dir (default <input>/figures)")
    ap.add_argument("--stats", default=None,
                    help="statistics output dir (default <input>/statistics)")
    ap.add_argument("--phases", default=None,
                    help="comma-separated phase boundaries, "
                         "e.g. '50,150,300,450'")
    args = ap.parse_args()

    input_dir = args.input
    fig_dir = args.output or os.path.join(input_dir, "figures")
    stat_dir = args.stats or os.path.join(input_dir, "statistics")

    print(f"v0.5.2 analysis")
    print(f"  input : {input_dir}")
    print(f"  figures: {fig_dir}")
    print(f"  stats : {stat_dir}")

    # phases: CLI > config file > default
    phases = None
    if args.phases:
        boundaries = [int(x) for x in args.phases.split(",") if x.strip()]
        if boundaries:
            phases = [(1, boundaries[0])] + [
                (boundaries[i] + 1, boundaries[i + 1])
                for i in range(len(boundaries) - 1)]
    if phases is None:
        cfg_path = os.path.join("config", "phase5_2_v052.yaml")
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            bounds = cfg.get("phase_boundaries") or []
            if bounds:
                phases = [(1, int(bounds[0]))] + [
                    (int(bounds[i]) + 1, int(bounds[i + 1]))
                    for i in range(len(bounds) - 1)]
        except Exception:  # noqa: BLE001
            pass
    if phases is None:
        phases = list(DEFAULT_PHASES)

    all_data = load_all(input_dir)
    conds = conditions_present(all_data)
    print(f"  conditions found: {conds if conds else '(none)'}")

    if not all_data:
        print("\nNo experiment data found. Nothing to analyze.")
        return

    n_seed_total = sum(len(v) for v in all_data.values())
    print(f"  total (condition, seed) runs: {n_seed_total}")

    print("\n[1/4] 12 long-term curves (§28)...")
    make_long_term_curves(all_data, fig_dir)

    print("\n[2/4] 4-phase analysis (§25)...")
    phase_analysis(all_data, phases, stat_dir)

    print("\n[3/4] statistical tests (§43/§44)...")
    statistical_tests(all_data, stat_dir)

    print("\n[4/4] experience absorption...")
    absorption_analysis(all_data, fig_dir)

    print("\nDone! Figures in", fig_dir)
    print("Statistics in", stat_dir)


if __name__ == "__main__":
    main()
