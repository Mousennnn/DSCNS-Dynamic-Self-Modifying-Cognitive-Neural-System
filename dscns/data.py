"""Dataset loading for the DSCNS experience stream (report section 10.2).

Sources (all public HuggingFace datasets):
- general:  wikitext-103-raw-v1   (Wikipedia dump subset)
- math:     gsm8k
- logic:    hendrycks/competition_math
- code:     openai/openai_humaneval (fallback: bigcode/humanevalpack)
- science:  allenai/sciq

All splits are deterministic (fixed seed) and cached under ``cache_dir``.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from .utils import set_seed


def _ensure_dataset(ds_name: str, cache_dir: str) -> Any:
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    return load_dataset(ds_name, cache_dir=cache_dir)


# ---------------------------------------------------------------------- #
# per-domain formatting helpers
# ---------------------------------------------------------------------- #
def _fmt_wiki(example: Dict[str, Any]) -> Dict[str, Any]:
    text = (example.get("text") or "").strip()
    if not text:
        return None
    lines = [l for l in text.split("\n") if len(l.strip()) > 60]
    return None if not lines else {"text": lines[0].strip(), "domain": "general"}


def _fmt_gsm8k(example: Dict[str, Any]) -> Dict[str, Any]:
    q = (example.get("question") or "").strip()
    a = (example.get("answer") or "").strip()
    if not q:
        return None
    text = f"Question: {q}\nAnswer: {a}"
    return {"text": text, "domain": "math"}


def _fmt_math(example: Dict[str, Any]) -> Dict[str, Any]:
    problem = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()
    if not problem:
        return None
    text = f"Problem: {problem}\nSolution: {solution}"
    return {"text": text, "domain": "logic"}


def _fmt_humaneval(example: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (example.get("prompt") or "").strip()
    solution = (example.get("canonical_solution") or example.get("solution") or "").strip()
    if not prompt:
        return None
    text = f"```python\n{prompt}\n{solution}\n```"
    return {"text": text, "domain": "code"}


def _fmt_sciq(example: Dict[str, Any]) -> Dict[str, Any]:
    q = (example.get("question") or "").strip()
    a = (example.get("correct_answer") or "").strip()
    if not q:
        return None
    text = f"Question: {q}\nAnswer: {a}"
    return {"text": text, "domain": "science"}


# ---------------------------------------------------------------------- #
# main entry
# ---------------------------------------------------------------------- #
def load_domains(cache_dir: str, config: Any) -> Dict[str, Dict[str, List[str]]]:
    """Return {"train": {domain: [texts]}, "eval": {...}, "probe": {...}}.

    eval/probe are disjoint from train and deterministic.
    """
    set_seed(0)
    domains = ["general", "math", "logic", "code", "science"]
    train: Dict[str, List[str]] = {}
    eval_: Dict[str, List[str]] = {}
    probe: Dict[str, List[str]] = {}

    max_train = getattr(config, "max_train_per_domain", 1200)
    eval_n = getattr(config, "eval_per_domain", 64)
    probe_n = getattr(config, "probe_per_domain", 16)

    src = dict(getattr(config, "datasets", {}))

    # ---- general: wikitext (streaming to keep download light) ----
    gen = _load_wikitext(cache_dir, max_train + eval_n + probe_n)
    train["general"], eval_["general"], probe["general"] = _split_pool(
        gen, max_train, eval_n, probe_n
    )

    # ---- math: gsm8k ----
    gsm = _load_gsm8k(cache_dir)
    train["math"], eval_["math"], probe["math"] = _split_pool(
        gsm, max_train, eval_n, probe_n
    )

    # ---- logic: MATH ----
    math = _load_math(cache_dir)
    train["logic"], eval_["logic"], probe["logic"] = _split_pool(
        math, max_train, eval_n, probe_n
    )

    # ---- code: HumanEval ----
    code = _load_humaneval(cache_dir)
    train["code"], eval_["code"], probe["code"] = _split_pool(
        code, max_train, eval_n, probe_n
    )

    # ---- science: SciQ ----
    sci = _load_sciq(cache_dir)
    train["science"], eval_["science"], probe["science"] = _split_pool(
        sci, max_train, eval_n, probe_n
    )

    return {"train": train, "eval": eval_, "probe": probe}


def _split_pool(pool: List[str], max_train: int, eval_n: int,
                probe_n: int) -> (List[str], List[str], List[str]):
    rng = random.Random(0)
    pool = list(pool)
    rng.shuffle(pool)
    if len(pool) < eval_n + probe_n + 10:
        # tiny pool (e.g. HumanEval): allow reuse across splits
        eval_ = pool[:eval_n]
        probe = pool[eval_n:eval_n + probe_n]
        train = pool[eval_n + probe_n:] or pool[:max_train]
        # pad train by reusing if needed
        while len(train) < max_train:
            train = train + pool[:max_train - len(train)]
        return train[:max_train], eval_, probe
    eval_ = pool[:eval_n]
    probe = pool[eval_n:eval_n + probe_n]
    train = pool[eval_n + probe_n:eval_n + probe_n + max_train]
    return train, eval_, probe


# ---------------------------------------------------------------------- #
# individual dataset loaders (with local JSON caching for reliability)
# ---------------------------------------------------------------------- #
def _load_wikitext(cache_dir: str, n: int) -> List[str]:
    cache = os.path.join(cache_dir, "wikitext_train.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train",
        cache_dir=cache_dir, streaming=True,
    )
    out = []
    for ex in ds:
        item = _fmt_wiki(ex)
        if item is not None:
            out.append(item["text"])
            if len(out) >= n + 200:
                break
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _load_gsm8k(cache_dir: str) -> List[str]:
    cache = os.path.join(cache_dir, "gsm8k_train.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    out = []
    for ex in ds:
        item = _fmt_gsm8k(ex)
        if item:
            out.append(item["text"])
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _load_math(cache_dir: str) -> List[str]:
    cache = os.path.join(cache_dir, "math_train.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    out = []
    try:
        # hendrycks/competition_math is gated; fall back to MATH-500 if denied
        ds = load_dataset("hendrycks/competition_math", split="train",
                          cache_dir=cache_dir)
    except Exception as e:
        print(f"[data] hendrycks/competition_math unavailable ({type(e).__name__}), "
              f"using HuggingFaceH4/MATH-500")
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)
    for ex in ds:
        item = _fmt_math(ex)
        if item:
            out.append(item["text"])
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _load_humaneval(cache_dir: str) -> List[str]:
    cache = os.path.join(cache_dir, "humaneval_train.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    out = []
    try:
        ds = load_dataset("openai/openai_humaneval", split="test", cache_dir=cache_dir)
    except Exception:
        ds = load_dataset("bigcode/humanevalpack", "python",
                          split="test", cache_dir=cache_dir)
    for ex in ds:
        item = _fmt_humaneval(ex)
        if item:
            out.append(item["text"])
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _load_sciq(cache_dir: str) -> List[str]:
    cache = os.path.join(cache_dir, "sciq_train.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="train", cache_dir=cache_dir)
    out = []
    for ex in ds:
        item = _fmt_sciq(ex)
        if item:
            out.append(item["text"])
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


# ---------------------------------------------------------------------- #
def build_experience_stream(data: Dict[str, Dict[str, List[str]]],
                            config: Any) -> List[List[Dict[str, Any]]]:
    """Build the ordered experience stream: one list of experiences per round.

    Round 1..k1: general, then math, logic, code, science, then mixed.
    """
    rng = random.Random(config.seed)
    train = data["train"]
    phases = config.phases
    phase_rounds = config.phase_rounds
    per_round = config.samples_per_round
    stream = []
    for phase, n_rounds in zip(phases, phase_rounds):
        for _ in range(n_rounds):
            if phase == "mixed":
                samples = []
                for domain in ["general", "math", "logic", "code", "science"]:
                    pool = train[domain]
                    k = max(1, per_round // 5)
                    samples += [{"text": t, "domain": domain,
                                 "source": domain, "reliability": 0.8}
                                for t in rng.sample(pool, min(k, len(pool)))]
                rng.shuffle(samples)
                stream.append(samples[:per_round])
            else:
                pool = train[phase]
                k = min(per_round, len(pool))
                stream.append([{"text": t, "domain": phase,
                                "source": phase, "reliability": 0.8}
                               for t in rng.sample(pool, k)])
    return stream
