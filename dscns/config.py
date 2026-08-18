"""Configuration for DSCNS experiments (Phase 0 + Phase 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DSCNSConfig:
    # ---- base model (Phase 0) ----
    model_name: str = "models/hf/gpt2"  # local copy of gpt2 (124M)
    device: str = "cuda"
    torch_dtype: str = "float32"
    cache_dir: Optional[str] = None

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_lr: float = 5e-4

    # ---- data / stream ----
    data_cache: str = "data"
    max_train_per_domain: int = 1200
    eval_per_domain: int = 64
    probe_per_domain: int = 16
    exemplars_per_domain: int = 48
    samples_per_round: int = 32
    phases: List[str] = field(default_factory=lambda: [
        "general", "math", "logic", "code", "science", "mixed"])
    phase_rounds: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4])

    # ---- parsing / evaluation ----
    max_len: int = 192
    parse_batch: int = 8
    eval_batch: int = 8
    probe_batch: int = 8
    probe_size: int = 16

    # ---- verification ----
    conflict_threshold: float = 0.4
    trust_lr: float = 0.05
    acceptance_threshold: float = 0.25

    # ---- internalization ----
    internalization_tolerance: float = 0.02
    max_alpha: float = 1.0   # alpha ramp max (steps: 0.2..1.0 x base_lr)
    internalization_steps: int = 5

    # ---- meta ----
    buffer_capacity: int = 10000
    store_threshold: float = 0.30

    # ---- structure evolution (Phase 3) ----
    evolution_enabled: bool = False
    evolution_every: int = 4
    evolution_min_round: int = 6  # stabilization period before any evolution
    split_diversity_threshold: float = 0.8
    merge_overlap_threshold: float = 0.97
    merge_co_activation_threshold: int = 8
    merge_similarity_threshold: float = 0.9

    # ---- learned structural self-adaptation (Phase 4) ----
    # controller: "rule" (Phase 3 flow) | "single_rule" | "learned" | "none"
    evolution_controller: str = "rule"
    learned_warmup_rounds: int = 8       # Stage A: rule-driven imitation
    policy_lr: float = 3e-4
    policy_hidden: int = 64
    policy_temperature: float = 0.8
    policy_epsilon: float = 0.15         # Stage B exploration rate
    modification_budget_max: int = 8     # hard cap on network count
    adaptation_window: int = 3           # rounds over which a change is judged
    modification_tolerance: float = 0.02 # probe drop allowed before rollback
    reward_lambda_forgetting: float = 0.5
    reward_lambda_params: float = 0.3
    reward_lambda_compute: float = 0.1
    reward_lambda_instability: float = 0.3
    total_rounds: int = 16               # expected stream length (state feature)

    # ---- intrinsic parameter self-modification (Phase 5) ----
    # P5-B: theta -> h -> delta_theta -> theta'  (IntrinsicPlasticityModule)
    enable_plasticity: bool = False      # attach the plasticity module to networks
    plasticity_mode: str = "modification"  # "modification" (P5-B) | "modulation" (P5-A)
    plasticity_interval: int = 4         # external trigger: every N grad steps
    plasticity_alpha: float = 0.01       # theta' = theta + alpha * delta_theta
    min_delta_threshold: float = 1e-6    # skip near-zero deltas
    plasticity_hidden_dim: int = 768     # GPT-2 hidden dim
    meta_dim: int = 32                   # self-state meta vector dim (s_t)
    plasticity_rank: int = 8             # low-rank delta generation rank
    plasticity_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    modulation_strength_init: float = 0.05  # learnable global modulation strength init
    plasticity_lr: float = 1e-5          # plasticity module LR (P5-C)
    train_plasticity: bool = False       # P5-B off / P5-C on
    plasticity_train_threshold: int = 10 # start training after this many success cases
    plasticity_train_batches: int = 5
    plasticity_train_batch_size: int = 4
    min_memory_size: int = 10
    max_memory_size: int = 100
    adaptation_steps: int = 3            # short adaptation before reward (P5-C)
    # state-component ablation (which inputs feed P_phi)
    use_hidden: bool = True              # h_t
    use_param_stats: bool = True         # stats(theta_t)
    use_meta: bool = True                # s_t
    # validation / safety (experiment controller, not model mechanism)
    quick_validation_samples: int = 8
    validation_loss_margin: float = 0.5  # nats; accept if loss_after < loss_before + margin
    validation_perplexity_cap: float = 100.0
    max_param_change_ratio: float = 0.1
    rollback_on_failure: bool = True
    save_snapshots: bool = True
    run_negative_controls: bool = True   # random / constant / shuffled delta arms
    task_lr: float = 5e-5                # task-learning LR used by the P5 loop

    # ---- experiment ----
    num_networks: int = 5
    seed: int = 42
    eval_every: int = 1
    with_generation_eval: bool = False
    train_steps_per_round: int = 8  # compute budget for Control mode
    max_grad_steps_per_round: int = 8  # compute budget for Exp1/Exp2 (parity)

    # ---- data sources ----
    datasets: Dict[str, str] = field(default_factory=lambda: {
        "general": "wikitext",          # wikitext-103-raw-v1 (Wikipedia dump)
        "math": "gsm8k",                # GSM8K
        "logic": "math",                # hendrycks/competition_math
        "code": "humaneval",            # openai/openai_humaneval
        "science": "sciq",              # allenai/sciq
    })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DSCNSConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name, "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha, "lora_lr": self.lora_lr,
            "samples_per_round": self.samples_per_round,
            "phases": self.phases, "phase_rounds": self.phase_rounds,
            "num_networks": self.num_networks, "seed": self.seed,
            "internalization_tolerance": self.internalization_tolerance,
            "max_alpha": self.max_alpha,
            "internalization_steps": self.internalization_steps,
            "acceptance_threshold": self.acceptance_threshold,
            "conflict_threshold": self.conflict_threshold,
            "evolution_enabled": self.evolution_enabled,
        }
