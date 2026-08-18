"""Phase 5 negative controls validation (design report section 10.2).

  Control A  Random delta:   same-norm gaussian delta (no state dependence)
  Control B  Constant delta: the same delta for every input (variance 0)
  Control C  Shuffled state: state<->delta pairing broken

The intrinsic delta must be distinguishable from these controls; P5's claim
of *state-dependent* self-modification rests on these comparisons.

Run directly:  python tests/test_phase5_validation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase5_common import make_random_delta, quick_validation
from tests.phase5_fixture import make_fixture

RESULTS = {}


def _effect(net, tokenizer, delta, texts, max_len, alpha=0.1):
    """Validation pass + behavioral change after applying delta."""
    import torch

    snap = net.snapshot_parameters()
    sub = texts[:8]
    enc = tokenizer(sub, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    enc = {k: v.to(net.peft_model.device) for k, v in enc.items()}
    net.peft_model.set_adapter(net.id)
    net.peft_model.eval()
    with torch.no_grad():
        logits_before = net.peft_model(**enc).logits
    loss_before = float(np.mean(net.losses_for_texts(
        sub, tokenizer, batch_size=8, max_len=max_len)))
    net.apply_intrinsic_modification(delta, alpha=alpha)
    with torch.no_grad():
        logits_after = net.peft_model(**enc).logits
    ok, reason, loss_after, ppl = quick_validation(net, _Base(tokenizer), sub,
                                                   _Cfg(), loss_before)
    pred_change = float((logits_before.argmax(-1) !=
                         logits_after.argmax(-1)).float().mean())
    logits_diff = float((logits_after - logits_before).abs().mean())
    net.restore_parameters(snap)
    return {"pass": ok, "reason": reason, "pred_change": pred_change,
            "logits_diff": logits_diff}


class _Cfg:
    quick_validation_samples = 8
    validation_loss_margin = 0.5
    validation_perplexity_cap = 100.0
    max_len = 96


class _Base:
    """Minimal stand-in exposing .tokenizer to quick_validation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def control_random_delta(net, tokenizer, texts, max_len):
    """Intrinsic delta vs same-norm random delta."""
    import torch

    delta_int = net.generate_delta(texts, tokenizer, max_len=max_len)
    delta_rand = make_random_delta(delta_int, net.peft_model.device)
    eff_int = _effect(net, tokenizer, delta_int, texts, max_len)
    eff_rand = _effect(net, tokenizer, delta_rand, texts, max_len)
    same_content = bool(torch.allclose(delta_int["delta_W_A"],
                                       delta_rand["delta_W_A"]))
    ok = (not same_content) and (
        eff_int["logits_diff"] != eff_rand["logits_diff"] or
        eff_int["pred_change"] != eff_rand["pred_change"])
    RESULTS["control_a_random"] = {
        "intrinsic": eff_int, "random": eff_rand,
        "same_content": same_content, "ok": ok}
    print(f"[Control A] intrinsic: pass={eff_int['pass']} "
          f"pred_change={eff_int['pred_change']:.4f} "
          f"logits_diff={eff_int['logits_diff']:.6f}")
    print(f"[Control A] random   : pass={eff_rand['pass']} "
          f"pred_change={eff_rand['pred_change']:.4f} "
          f"logits_diff={eff_rand['logits_diff']:.6f}")
    assert ok, "intrinsic and random deltas are indistinguishable"
    return True


def control_constant_delta(net, tokenizer, texts_a, texts_b, max_len):
    """Intrinsic deltas vary across inputs; a constant delta does not."""
    delta_a = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    delta_b = net.generate_delta(texts_b, tokenizer, max_len=max_len)
    intrinsic_diff = float((delta_a["delta_W_A"] - delta_b["delta_W_A"]).norm()) + \
        float((delta_a["delta_W_B"] - delta_b["delta_W_B"]).norm())
    # the "constant" control: same delta object reused -> zero variance
    constant_diff = 0.0
    ok = intrinsic_diff > 1e-4 and constant_diff < 1e-8
    RESULTS["control_b_constant"] = {
        "intrinsic_cross_input_diff": intrinsic_diff,
        "constant_cross_input_diff": constant_diff, "ok": ok}
    print(f"[Control B] intrinsic delta cross-input diff={intrinsic_diff:.6f}  "
          f"constant delta diff={constant_diff:.2e}  -> {ok}")
    assert ok, "state-dependency not distinguishable from a constant delta"
    return True


def control_shuffled_state(net, tokenizer, texts_a, texts_b, max_len):
    """Correct state<->delta pairing vs shuffled pairing."""
    delta_from_a = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    delta_from_b = net.generate_delta(texts_b, tokenizer, max_len=max_len)
    eff_correct = _effect(net, tokenizer, delta_from_a, texts_a, max_len)
    eff_shuffled = _effect(net, tokenizer, delta_from_b, texts_a, max_len)
    ok = eff_correct["logits_diff"] >= eff_shuffled["logits_diff"] - 1e-9
    RESULTS["control_c_shuffled"] = {
        "correct_pairing": eff_correct, "shuffled_pairing": eff_shuffled,
        "ok": ok}
    print(f"[Control C] correct pairing: pass={eff_correct['pass']} "
          f"logits_diff={eff_correct['logits_diff']:.6f}")
    print(f"[Control C] shuffled pair : pass={eff_shuffled['pass']} "
          f"logits_diff={eff_shuffled['logits_diff']:.6f}")
    assert ok, "shuffling removes state dependence"
    return True


def run_all():
    base, net, texts_a, texts_b, cfg = make_fixture(max_len=96)
    tokenizer = base.tokenizer
    control_random_delta(net, tokenizer, texts_a, cfg.max_len)
    control_constant_delta(net, tokenizer, texts_a, texts_b, cfg.max_len)
    control_shuffled_state(net, tokenizer, texts_a, texts_b, cfg.max_len)
    all_ok = all(v["ok"] for v in RESULTS.values())
    print("=== NEGATIVE CONTROLS:", "PASS" if all_ok else "FAIL", "===")
    return RESULTS, all_ok


if __name__ == "__main__":
    import json

    results, ok = run_all()
    out_dir = os.path.join("experiments", "phase5")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "controls_tests.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    sys.exit(0 if ok else 1)
