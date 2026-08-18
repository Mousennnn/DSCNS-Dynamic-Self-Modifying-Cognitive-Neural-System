"""Shared test fixture for Phase 5 validation suites.

Builds one CognitiveNetwork with an IntrinsicPlasticityModule on the local
GPT-2 copy, plus two small input-text groups (different domains) used by
the state-dependency tests.  Data is read from the local HF cache JSONs so
the suites run offline.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dscns.config import DSCNSConfig
from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
from dscns.memory import MemorySystem
from dscns.networks import CognitiveNetwork

DATA_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "hf")


CACHE_NAMES = {
    "general": "wikitext", "math": "gsm8k", "logic": "math",
    "code": "humaneval", "science": "sciq",
}


def _texts(name: str, n: int = 8) -> list:
    fname = CACHE_NAMES.get(name, name)
    with open(os.path.join(DATA_CACHE, f"{fname}_train.json"),
              "r", encoding="utf-8") as f:
        pool = json.load(f)
    return [str(t) for t in pool[:n]]


def make_fixture(max_len: int = 96, plasticity_rank: int = 8,
                 modulation_strength_init: float = 0.05,
                 enable_plasticity: bool = True):
    from scripts.common import make_base_model

    cfg = DSCNSConfig(
        model_name=os.path.join("models", "hf", "gpt2"),
        cache_dir="models/hf",
        max_len=max_len,
        lora_r=16,
        plasticity_rank=plasticity_rank,
        modulation_strength_init=modulation_strength_init,
    )
    base = make_base_model(cfg, tag="p5_test")
    base.add_adapter("N1")
    plasticity = None
    if enable_plasticity:
        plasticity = IntrinsicPlasticityModule(
            hidden_dim=cfg.plasticity_hidden_dim,
            adapter_dim=cfg.lora_r,
            meta_dim=cfg.meta_dim,
            plasticity_rank=plasticity_rank,
            modulation_strength_init=modulation_strength_init,
        ).to(base.peft_model.device)
    net = CognitiveNetwork(
        net_id="N1", name="WorldKnowledge", domain="general",
        peft_model=base.peft_model, memory=MemorySystem(),
        base_lr=5e-5, plasticity=plasticity,
        plasticity_cfg={"meta_dim": cfg.meta_dim},
    )
    net.set_trainable(False)
    texts_a = _texts("wikitext", 8)
    texts_b = _texts("code", 8)
    return base, net, texts_a, texts_b, cfg
