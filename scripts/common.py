"""Shared experiment harness pieces for DSCNS Phase 1-3 runs."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dscns.base_model import BaseLanguageModel
from dscns.config import DSCNSConfig
from dscns.data import build_experience_stream, load_domains
from dscns.memory import MemorySystem
from dscns.networks import CognitiveNetwork
from dscns.system import DSCNSSystem
from dscns.utils import set_seed

DOMAINS = ["general", "math", "logic", "code", "science"]


def make_config(cfg_path: Optional[str] = None, **overrides) -> DSCNSConfig:
    import os

    # memory-friendly defaults for the 8GB GPU
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")
    cfg = DSCNSConfig()
    if cfg_path and os.path.exists(cfg_path):
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = DSCNSConfig.from_dict(yaml.safe_load(f))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def make_base_model(config: DSCNSConfig, tag: str) -> BaseLanguageModel:
    cache = os.path.join(config.cache_dir or "models", "hf")
    os.makedirs(cache, exist_ok=True)
    model_id = config.model_name
    # prefer a local copy of the model over a Hub download
    if not os.path.isdir(model_id):
        local = os.path.join("models", "hf", model_id)
        if os.path.isdir(local):
            model_id = local
    if os.path.isdir(model_id):
        os.environ["HF_HUB_OFFLINE"] = "1"
    return BaseLanguageModel(
        model_name=model_id,
        device=config.device,
        torch_dtype=config.torch_dtype,
        max_len=config.max_len,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        cache_dir=cache,
    )


def prepare_data(config: DSCNSConfig) -> Dict[str, Dict[str, List[str]]]:
    cache = os.path.join(config.data_cache, "hf")
    os.makedirs(cache, exist_ok=True)
    return load_domains(cache, config)


def build_exemplars(data, config: DSCNSConfig) -> Dict[str, List[str]]:
    return {d: texts[: config.exemplars_per_domain]
            for d, texts in data["train"].items()}


def build_system(config: DSCNSConfig, base: BaseLanguageModel,
                 data: Dict[str, Dict[str, List[str]]]) -> DSCNSSystem:
    exemplars = build_exemplars(data, config)
    probe = {d: t[: config.probe_per_domain] for d, t in data["probe"].items()}
    system = DSCNSSystem(base, config, exemplars, probe, seed=config.seed)
    system.set_eval_sets({d: t for d, t in data["eval"].items()})
    return system


# ---------------------------------------------------------------------- #
# evaluation helpers
# ---------------------------------------------------------------------- #
def eval_per_domain_loss(system_or_net: Any, eval_sets: Dict[str, List[str]],
                         base: Optional[BaseLanguageModel] = None,
                         eval_size: int = 64) -> Dict[str, float]:
    """exp(-loss) per domain (best network for systems)."""
    if isinstance(system_or_net, DSCNSSystem):
        return system_or_net.best_domain_performance(eval_sets, eval_size)
    # single network / control learner
    perfs = {}
    for domain, texts in eval_sets.items():
        perfs[domain] = system_or_net.evaluate_texts(
            texts[:eval_size], base.tokenizer, batch_size=8
        )
    return perfs


def generation_eval(model_or_system: Any, eval_sets: Dict[str, List[str]],
                    base: BaseLanguageModel, domains: List[str] = ("science", "math"),
                    n: int = 24) -> Dict[str, float]:
    """Exact-answer match accuracy via greedy generation.

    Expects texts formatted as 'Question: ...\\nAnswer: ...'.
    For DSCNSSystem, the most relevant network answers per domain.
    """
    from dscns.system import DSCNSSystem

    DOMAIN_TO_NET = {"general": "world", "math": "math", "logic": "logic",
                     "code": "language", "science": "verification"}
    net_by_domain = {}
    if isinstance(model_or_system, DSCNSSystem):
        for net in model_or_system.networks.values():
            net_by_domain[net.domain] = net.id

    results = {}
    for domain in domains:
        texts = eval_sets.get(domain, [])[:n]
        if not texts:
            continue
        prompts, golds = [], []
        for t in texts:
            parts = t.split("\nAnswer: ")
            prompts.append(parts[0] + "\nAnswer:")
            golds.append(parts[1].strip() if len(parts) > 1 else "")
        net_id = net_by_domain.get(domain)
        if net_id:
            base.peft_model.set_adapter(net_id)
        gens = base.generate(prompts, max_new_tokens=48)
        correct = 0
        for g, gold in zip(gens, golds):
            if _exact_match(g, gold, domain):
                correct += 1
        results[domain] = correct / len(gens)
    return results


def _exact_match(generated: str, gold: str, domain: str) -> bool:
    import re

    g = generated.strip().lower()
    gold_l = gold.strip().lower()
    if domain == "math":
        # GSM8K gold answer ends with '#### <number>'
        m = re.search(r"####\s*([\d.,]+)", gold_l)
        target = m.group(1).replace(",", "") if m else gold_l
        return target in re.sub(r"\s+", "", g)
    # science: answer phrase contained in generation
    return gold_l[:40] in g


# ---------------------------------------------------------------------- #
# control (baseline) training
# ---------------------------------------------------------------------- #
def make_control_network(base: BaseLanguageModel) -> CognitiveNetwork:
    base.add_adapter("learner")
    net = CognitiveNetwork(
        net_id="learner", name="ControlLearner", domain="general",
        peft_model=base.peft_model, memory=MemorySystem(),
        domain_embedding=None, base_lr=5e-4,
    )
    net.set_trainable(True)
    return net


def control_train_step(net: CognitiveNetwork, texts: List[str],
                       base: BaseLanguageModel, lr: float = 5e-4,
                       batch_size: int = 8, max_len: int = 256) -> float:
    """One standard gradient step (sequential fine-tuning baseline)."""
    import torch

    net.set_trainable(True)
    base.peft_model.set_adapter("learner")
    base.peft_model.train()
    opt = net.get_optimizer()
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad()
    enc = base.tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len)
    enc = {k: v.to(base.device) for k, v in enc.items()}
    loss_sum = 0.0
    n_chunks = max(1, (len(texts) + batch_size - 1) // batch_size)
    for i in range(0, len(texts), batch_size):
        chunk = {k: v[i:i + batch_size] for k, v in enc.items()}
        out = base.peft_model(**chunk, labels=chunk["input_ids"])
        (out.loss / n_chunks).backward()
        loss_sum += float(out.loss.item())
        del out, chunk
    torch.nn.utils.clip_grad_norm_(
        [p for p in net._adapter_params() if p.grad is not None], max_norm=1.0
    )
    opt.step()
    base.peft_model.eval()
    return loss_sum / n_chunks


# ---------------------------------------------------------------------- #
# results persistence
# ---------------------------------------------------------------------- #
def save_results(out_dir: str, name: str, results: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def plot_performance_matrix(matrix: List[List[float]], domains: List[str],
                            out_path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Round")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30)
    ax.set_title(title)
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_curves(results: Dict[str, Any], out_dir: str, domains: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = np.asarray(results["performance_matrix"])
    rounds = range(1, pm.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, d in enumerate(domains):
        ax.plot(list(rounds), pm[:, j], marker="o", markersize=3, label=d)
    ax.set_xlabel("Round")
    ax.set_ylabel("Performance (exp(-loss))")
    ax.set_title("Per-domain performance over rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=140)
    plt.close(fig)
