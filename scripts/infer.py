"""v0.6.0 / Phase 6 Inference Pipeline.

Demonstrates the complete DSCNS self-modification pipeline:

    Input → Internal State → Error Detection → Experience Retrieval
    → Policy Selection → Modification Proposal → Applied Δθ → Output

Three modes:
    --mode baseline     : frozen parameters (no modification)
    --mode self-modify  : one runtime modification
    --mode relay        : load from relay checkpoint
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BATCH_SIZE = 8
MAX_LEN = 192


def load_model(checkpoint_path=None, device="cuda"):
    """Load the base model and optionally a checkpoint."""
    from scripts.common import make_base_model, make_config
    from scripts.phase5_common import build_phase5_networks
    from dscns.intrinsic_plasticity import IntrinsicPlasticityModule

    p5_cfg = make_config(cfg_path="config/phase5.yaml")
    p5_cfg.num_networks = 1
    p5_cfg.seed = 42

    base = make_base_model(p5_cfg, tag="inference")
    networks = build_phase5_networks(base, p5_cfg)
    net = networks[0]

    net.plasticity = IntrinsicPlasticityModule(
        hidden_dim=768, adapter_dim=16, meta_dim=32, plasticity_rank=8,
        p51=True, m_min=0.02, m_max=1.0, m_init_bias=-3.0,
        error_dim=32, num_target_groups=3).to(base.device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=base.device)
        if "plasticity" in state:
            net.plasticity.load_state_dict(state["plasticity"], strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")

    return base, net


def compute_internal_state(net, texts, tokenizer, max_len=MAX_LEN):
    """Compute internal state from input texts."""
    with torch.no_grad():
        out_h = net.generate_delta(texts, tokenizer, max_len=max_len, grad_enabled=False)
    pooled = out_h["components"]["pooled_h"].detach()
    if pooled.dim() == 3:
        pooled = pooled.mean(1)
    pooled = pooled.mean(0, keepdim=True)
    return {
        "pooled_hidden": pooled,
        "delta_W_A": out_h["delta_W_A"],
        "delta_W_B": out_h["delta_W_B"],
    }


def detect_error(net, state, prev_loss=None, tokenizer=None):
    """Detect error in current state."""
    from dscns.error_correction import ErrorState
    if prev_loss is not None:
        # compute loss on texts
        loss = compute_loss_on_texts(net, texts=None, tokenizer=tokenizer)
    else:
        loss = 0.0
    return ErrorState(
        task_delta=0.0,
        probe_delta=loss - (prev_loss or 0.0),
        logit_delta=0.0, entropy_delta=0.0,
        parameter_drift=0.0, prev_target=0, prev_magnitude=0.0)


def compute_loss_on_texts(net, texts, tokenizer):
    """Compute CE loss on texts."""
    if texts is None or not texts:
        return 4.0  # default
    import torch.nn.functional as F
    net.peft_model.eval()
    enc = tokenizer(texts[:4], return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(net.peft_model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = net.peft_model(**enc, labels=enc["input_ids"])
    return float(out.loss.item())


def run_inference(texts, checkpoint=None, mode="baseline", device="cuda"):
    """Run the full inference pipeline.

    Args:
        texts: input texts.
        checkpoint: path to model checkpoint.
        mode: "baseline", "self-modify", or "relay".
        device: compute device.

    Returns:
        Dict with pipeline trace.
    """
    base, net = load_model(checkpoint, device)
    tokenizer = base.tokenizer

    trace = {
        "mode": mode,
        "input_texts": texts[:3],
        "pipeline": [],
    }

    # Step 1: Baseline evaluation
    baseline_loss = compute_loss_on_texts(net, texts, tokenizer)
    trace["baseline_loss"] = baseline_loss
    trace["pipeline"].append({
        "step": 1, "name": "Baseline",
        "description": f"CE loss = {baseline_loss:.4f}",
    })

    if mode == "baseline":
        trace["final_loss"] = baseline_loss
        trace["modification_applied"] = False
        return trace

    # Step 2: Compute internal state
    state = compute_internal_state(net, texts, tokenizer)
    trace["pipeline"].append({
        "step": 2, "name": "Internal State",
        "description": f"hidden_norm={state['pooled_hidden'].norm():.4f}",
    })

    # Step 3: Generate modification proposal
    with torch.no_grad():
        proposal = net.plasticity.generate_proposal(
            state["pooled_hidden"].unsqueeze(1),
            net._current_params_tensors(),
            net._get_meta_info(32),
            error_state=None, memory_z=None, mask=None)
    trace["pipeline"].append({
        "step": 3, "name": "Modification Proposal",
        "description": f"target={proposal.get('target_group', 0)}, "
                       f"magnitude={proposal.get('magnitude', 0):.4f}",
    })

    # Step 4: Apply modification
    if mode == "self-modify":
        before_snap = net.snapshot_parameters()
        applied = net.apply_self_modification(proposal, alpha=0.01)
        after_loss = compute_loss_on_texts(net, texts, tokenizer)
        trace["after_loss"] = after_loss
        trace["delta_loss"] = after_loss - baseline_loss
        trace["modification_applied"] = applied
        trace["pipeline"].append({
            "step": 4, "name": "Applied Modification",
            "description": f"delta_loss={after_loss - baseline_loss:.4f}, "
                           f"applied={applied}",
        })
    else:
        trace["final_loss"] = baseline_loss
        trace["modification_applied"] = False

    trace["final_loss"] = trace.get("after_loss", baseline_loss)
    return trace


def main():
    ap = argparse.ArgumentParser(description="DSCNS v0.6.0 Inference")
    ap.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    ap.add_argument("--mode", choices=["baseline", "self-modify", "relay"],
                    default="baseline")
    ap.add_argument("--input", default=None, help="Input text or JSON file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None, help="Output JSON path")
    args = ap.parse_args()

    # default input texts
    if args.input:
        if os.path.exists(args.input):
            with open(args.input) as f:
                data = json.load(f)
            texts = data if isinstance(data, list) else [data]
        else:
            texts = [args.input]
    else:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In mathematics, the Pythagorean theorem states that a squared plus b squared equals c squared.",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        ]

    result = run_inference(texts, args.checkpoint, args.mode, args.device)

    # print trace
    print("\n" + "=" * 60)
    print(f"DSCNS v0.6.0 Inference — Mode: {result['mode']}")
    print("=" * 60)
    print(f"\nInput: {result['input_texts'][0][:80]}...")
    print(f"\nPipeline:")
    for step in result["pipeline"]:
        print(f"  Step {step['step']}: {step['name']} — {step['description']}")
    print(f"\nFinal Loss: {result['final_loss']:.4f}")
    print(f"Modification Applied: {result.get('modification_applied', False)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
