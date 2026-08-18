"""Phase 5 smoke test: intrinsic plasticity machinery end-to-end (fast)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dscns.config import DSCNSConfig
from dscns.intrinsic_plasticity import IntrinsicPlasticityModule
from dscns.plasticity_trainer import PlasticityTrainer
from scripts.phase5_common import (apply_plasticity_step, make_random_delta,
                                   quick_validation, train_round_step)


def _cfg():
    return DSCNSConfig(
        model_name=os.path.join("models", "hf", "gpt2"),
        cache_dir="models/hf",
        max_len=96,
        lora_r=16,
        task_lr=5e-5,
        plasticity_interval=2,
        plasticity_alpha=0.01,
        min_delta_threshold=1e-6,
        plasticity_rank=8,
        meta_dim=32,
        modulation_strength_init=0.05,
        plasticity_train_batches=2,
        plasticity_train_batch_size=2,
        min_memory_size=1,
        max_memory_size=10,
        plasticity_train_threshold=1,
        quick_validation_samples=4,
    )


def main():
    from scripts.common import make_base_model, prepare_data

    cfg = _cfg()
    data = prepare_data(cfg)
    base = make_base_model(cfg, tag="smoke_p5")
    base.add_adapter("N1")
    base.add_adapter("N2")
    from dscns.memory import MemorySystem
    from dscns.networks import CognitiveNetwork

    nets = []
    for i, (nid, dom) in enumerate([("N1", "general"), ("N2", "code")]):
        plasticity = IntrinsicPlasticityModule(
            hidden_dim=cfg.plasticity_hidden_dim, adapter_dim=cfg.lora_r,
            meta_dim=cfg.meta_dim, plasticity_rank=cfg.plasticity_rank,
            modulation_strength_init=cfg.modulation_strength_init,
        ).to(base.peft_model.device)
        net = CognitiveNetwork(
            net_id=nid, name=f"Smoke{i}", domain=dom,
            peft_model=base.peft_model, memory=MemorySystem(),
            base_lr=cfg.task_lr, plasticity=plasticity,
            plasticity_cfg={"meta_dim": cfg.meta_dim},
        )
        net.data_domain = dom
        net.set_trainable(False)
        nets.append(net)

    texts_a = data["train"]["general"][:8]
    texts_b = data["train"]["code"][:8]

    # [1] meta info shape
    m = nets[0]._get_meta_info(cfg.meta_dim)
    print("[1] meta shape:", tuple(m.shape))
    assert m.shape == (cfg.meta_dim,)

    # [2] delta generation + norms
    delta = nets[0].generate_delta(texts_a, base.tokenizer, max_len=cfg.max_len)
    nA, nB = float(delta["delta_W_A"].norm()), float(delta["delta_W_B"].norm())
    print(f"[2] delta norms W_A={nA:.6f} W_B={nB:.6f} "
          f"strength={delta['modulation_strength']:.4f}")
    assert nA > 1e-6 and nB > 1e-6

    # [3] state dependency
    delta_b = nets[0].generate_delta(texts_b, base.tokenizer, max_len=cfg.max_len)
    diff = float((delta["delta_W_A"] - delta_b["delta_W_A"]).norm()) + \
        float((delta["delta_W_B"] - delta_b["delta_W_B"]).norm())
    print(f"[3] cross-input delta diff: {diff:.6f}")
    assert diff > 1e-4

    # [4] apply + param transition + validation
    snap = nets[0].snapshot_parameters()
    nets[0].apply_intrinsic_modification(delta, alpha=1.0)
    chg = 0.0
    for n, p in nets[0].peft_model.named_parameters():
        if n in snap.get("lora_A", {}):
            chg += float((p.data - snap["lora_A"][n]).norm())
    nets[0].restore_parameters(snap)
    print(f"[4] param change after apply: {chg:.6f}")
    assert chg > 1e-6

    # [5] full plasticity step via protocol (intrinsic)
    batch = [{"text": t, "domain": "general"} for t in texts_a]
    rec = apply_plasticity_step(nets[0], base, batch, cfg, mode="intrinsic")
    print(f"[5] intrinsic step: applied={rec['applied']} accepted={rec['accepted']} "
          f"delta_norm={rec['delta_norm']:.6f} reason={rec.get('reason')}")
    assert rec["applied"] and rec["accepted"]

    # [6] random delta control
    rand = make_random_delta(delta, base.peft_model.device)
    rn = float(rand["delta_W_A"].norm()) + float(rand["delta_W_B"].norm())
    print(f"[6] random delta norm {rn:.6f} vs reference {nA + nB:.6f}")
    assert abs(rn - (nA + nB)) < 1e-3

    # [7] P5-C trainer with synthetic success cases (re-embedding replay)
    trainer = PlasticityTrainer(nets[1], cfg, base=base)
    import torch

    for _ in range(3):
        dd = nets[1].generate_delta(texts_b, base.tokenizer, max_len=cfg.max_len)
        trainer.record_success_case(texts=texts_b[:4], delta_params=dd, reward=0.05)
    loss = trainer.train_from_memory()
    print(f"[7] trainer loss: {loss}")
    assert loss is not None and np.isfinite(loss)

    # [8] train one round step (step counter + trigger cadence)
    before_steps = nets[0].step_count
    train_round_step(nets[0], base, texts_a, cfg)
    print(f"[8] step_count {before_steps} -> {nets[0].step_count}")
    assert nets[0].step_count > before_steps

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
