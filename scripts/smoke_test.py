"""Smoke test: model load, multi-adapter LoRA, one full cognitive round.

Run after dependencies are installed:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dscns.base_model import BaseLanguageModel
from dscns.config import DSCNSConfig
from dscns.memory import MemorySystem
from dscns.networks import CognitiveNetwork
from dscns.system import DSCNSSystem


def main():
    cfg = DSCNSConfig(
        model_name=os.path.join("models", "hf", "gpt2"),
        cache_dir="models/hf",
        num_networks=5,
        samples_per_round=8,
        max_grad_steps_per_round=4,
    )
    base = BaseLanguageModel(model_name=cfg.model_name, cache_dir=cfg.cache_dir)
    print("[1] base model loaded:", cfg.model_name,
          "| params:", base.num_parameters() // 1_000_000, "M")

    # ---- multi-adapter check ----
    base.add_adapter("N1")
    base.add_adapter("N2")
    base.peft_model.set_adapter("N2")
    texts = ["The capital of France is Paris.", "What is 2 + 2?"]
    out = base.generate(texts, max_new_tokens=8)
    print("[2] multi-adapter generate ok:", [o[:30] for o in out])

    # ---- tiny system round ----
    domain_exemplars = {
        "general": ["The capital of France is Paris."] * 4,
        "math": ["Question: 2+2=? Answer: 4"] * 4,
        "logic": ["Problem: solve x. Solution: x=1"] * 4,
        "code": ["def f(x): return x"] * 4,
        "science": ["Question: H2O? Answer: water"] * 4,
    }
    probe_sets = {d: t[:2] for d, t in domain_exemplars.items()}
    system = DSCNSSystem(base, cfg, domain_exemplars, probe_sets)
    print("[3] networks:", list(system.networks.keys()))

    experiences = [
        {"text": "The Eiffel Tower is in Paris, France.", "domain": "general",
         "source": "wiki", "reliability": 0.9},
        {"text": "Question: A train travels 60 km in 2 hours. What is its speed?\nAnswer: 30 km/h",
         "domain": "math", "source": "gsm8k", "reliability": 0.9},
        {"text": "def add(a, b):\n    return a + b", "domain": "code",
         "source": "humaneval", "reliability": 0.9},
        {"text": "Question: What is the chemical symbol for water?\nAnswer: H2O",
         "domain": "science", "source": "sciq", "reliability": 0.9},
    ]
    info = system.process_experiences(experiences)
    print("[4] decisions:",
          {k: v["action"] for k, v in info["decisions"].items()})
    print("[5] internalization:", {k: (v["success"], round(v["final_level"], 3))
                                   for k, v in info["internalization"].items()})
    perf = system.best_domain_performance({d: t for d, t in probe_sets.items()})
    print("[6] domain perf:", {k: round(v, 4) for k, v in perf.items()})
    print("[7] memory:", system.memory.snapshot())
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
