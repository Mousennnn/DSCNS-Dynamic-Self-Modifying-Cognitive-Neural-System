"""Phase 5 core validation: Tests 1-6 (design report section 10.1).

  Test 1  Delta Existence       ||delta_theta|| > 0, not numerical noise
  Test 2  State Dependency      different h -> different delta (with
                                determinism + random-h ablation controls)
  Test 3  Parameter Transition  theta_{t+1} != theta_t, ||theta'-theta||
  Test 4  Behavioral Change     F(x, theta') != F(x, theta)
  Test 5  Closed-loop Loop      theta0->h0->d0->theta1->h1->d1->... exists,
                                non-constant, non-diverging
  Test 6  Stability             20 consecutive steps: no NaN, bounded norms,
                                output distribution does not collapse

Run directly:  python tests/test_phase5_core.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.phase5_fixture import make_fixture

RESULTS = {}


def test_delta_existence(net, texts_a, tokenizer, max_len):
    delta = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    nA = float(delta["delta_W_A"].norm())
    nB = float(delta["delta_W_B"].norm())
    ok = nA > 1e-6 and nB > 1e-6
    RESULTS["test1_delta_existence"] = {
        "delta_W_A_norm": nA, "delta_W_B_norm": nB, "ok": ok}
    print(f"[Test 1] delta W_A norm={nA:.6f}  W_B norm={nB:.6f}  -> {ok}")
    assert ok, "delta_theta is zero / below noise threshold"
    return True


def test_state_dependency(net, texts_a, texts_b, tokenizer, max_len):
    delta1 = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    delta2 = net.generate_delta(texts_b, tokenizer, max_len=max_len)

    # determinism: same input twice -> identical delta
    delta1b = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    det_diff = float((delta1["delta_W_A"] - delta1b["delta_W_A"]).norm()) + \
        float((delta1["delta_W_B"] - delta1b["delta_W_B"]).norm())

    diff = float((delta1["delta_W_A"] - delta2["delta_W_A"]).norm()) + \
        float((delta1["delta_W_B"] - delta2["delta_W_B"]).norm())

    # ablation: random h (same params/meta) must give a different delta
    import torch

    fake_h = torch.randn_like(delta1["components"]["pooled_h"])
    delta_fake = net.plasticity.forward_from_pooled(
        fake_h, net._current_params_tensors(), delta1["meta_info"])
    ablation_diff = float((delta1["delta_W_A"] - delta_fake["delta_W_A"]).norm()) + \
        float((delta1["delta_W_B"] - delta_fake["delta_W_B"]).norm())

    ok = det_diff < 1e-8 and diff > 1e-4 and ablation_diff > 1e-4
    RESULTS["test2_state_dependency"] = {
        "determinism_diff": det_diff, "input_diff": diff,
        "random_h_ablation_diff": ablation_diff, "ok": ok}
    print(f"[Test 2] determinism={det_diff:.2e}  input_diff={diff:.6f}  "
          f"random_h_diff={ablation_diff:.6f}  -> {ok}")
    assert ok, "delta does not depend on internal state"
    return True


def test_parameter_transition(net, texts_a, tokenizer, max_len):
    snap = net.snapshot_parameters()
    delta = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    net.apply_intrinsic_modification(delta, alpha=1.0)
    change = 0.0
    for n, p in net.peft_model.named_parameters():
        if n in snap.get("lora_A", {}):
            change += float((p.data - snap["lora_A"][n]).norm())
        elif n in snap.get("lora_B", {}):
            change += float((p.data - snap["lora_B"][n]).norm())
    net.restore_parameters(snap)
    ok = change > 1e-6
    RESULTS["test3_parameter_transition"] = {"param_change": change, "ok": ok}
    print(f"[Test 3] ||theta' - theta|| = {change:.6f}  -> {ok}")
    assert ok, "parameters did not change after apply_intrinsic_modification"
    return True


def test_behavioral_change(net, texts_a, tokenizer, max_len):
    import torch

    net.peft_model.set_adapter(net.id)
    net.peft_model.eval()
    snap = net.snapshot_parameters()
    enc = tokenizer(texts_a, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    enc = {k: v.to(net.peft_model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits_before = net.peft_model(**enc).logits
    delta = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    net.apply_intrinsic_modification(delta, alpha=1.0)
    with torch.no_grad():
        logits_after = net.peft_model(**enc).logits
    logits_diff = float((logits_after - logits_before).abs().mean())
    pred_change = float((logits_before.argmax(-1) !=
                         logits_after.argmax(-1)).float().mean())
    net.restore_parameters(snap)
    ok = logits_diff > 1e-3
    RESULTS["test4_behavioral_change"] = {
        "logits_diff": logits_diff, "pred_change_rate": pred_change, "ok": ok}
    print(f"[Test 4] logits diff={logits_diff:.6f}  "
          f"pred change={pred_change:.2%}  -> {ok}")
    assert ok, "model behavior did not change after modification"
    return True


def test_modification_loop(net, texts_a, tokenizer, max_len, iterations=5):
    sequence = []
    for i in range(iterations):
        delta = net.generate_delta(texts_a, tokenizer, max_len=max_len)
        dnorm = float(delta["delta_W_A"].norm()) + \
            float(delta["delta_W_B"].norm())
        sequence.append({
            "iteration": i, "delta_norm": dnorm,
            "param_norm": float(net._current_params_tensors()["W_A"].norm()) +
                          float(net._current_params_tensors()["W_B"].norm()),
        })
        net.apply_intrinsic_modification(delta, alpha=0.1)
    norms = [m["delta_norm"] for m in sequence]
    ok = (all(n > 1e-6 for n in norms) and
          float(np.var(norms)) > 1e-8 and
          all(n < 100.0 for n in norms))
    RESULTS["test5_modification_loop"] = {
        "delta_norms": norms, "delta_variance": float(np.var(norms)), "ok": ok}
    print(f"[Test 5] deltas={[f'{n:.4f}' for n in norms]}  "
          f"var={float(np.var(norms)):.2e}  -> {ok}")
    assert ok, "closed loop missing / constant / diverging"
    return True


def test_stability(net, texts_a, tokenizer, max_len, steps=20):
    import torch
    import torch.nn.functional as F

    metrics = []
    for step in range(steps):
        delta = net.generate_delta(texts_a, tokenizer, max_len=max_len)
        net.apply_intrinsic_modification(delta, alpha=0.05)
        tensors = net._current_params_tensors()
        W = torch.cat([tensors["W_A"], tensors["W_B"]])
        param_norm = float(W.norm())
        has_nan = bool(torch.isnan(W).any())
        enc = tokenizer(texts_a[:4], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        enc = {k: v.to(net.peft_model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = net.peft_model(**enc).logits
        entropy = float(-(F.softmax(logits, dim=-1) *
                          F.log_softmax(logits, dim=-1)).sum(-1).mean())
        metrics.append({"step": step, "param_norm": param_norm,
                        "has_nan": has_nan, "entropy": entropy})
        assert not has_nan, f"NaN at step {step}"
        assert param_norm < 1000.0, f"explosion at step {step}"
        assert param_norm > 1e-3, f"collapse at step {step}"
        assert entropy > 0.1, f"output collapsed at step {step}"
    ok = True
    RESULTS["test6_stability"] = {
        "param_norms": [m["param_norm"] for m in metrics],
        "entropies": [m["entropy"] for m in metrics],
        "ok": ok}
    print(f"[Test 6] param_norm {metrics[0]['param_norm']:.3f} -> "
          f"{metrics[-1]['param_norm']:.3f}  "
          f"entropy {metrics[0]['entropy']:.3f} -> {metrics[-1]['entropy']:.3f}"
          f"  -> {ok}")
    return True


def test_modulation_p5a(net, texts_a, tokenizer, max_len):
    """P5-A: transient modulation changes computation, then restores."""
    import torch

    snap_before = net.snapshot_parameters()
    delta = net.generate_delta(texts_a, tokenizer, max_len=max_len)
    out = net.modulate_forward(texts_a, tokenizer, delta, alpha=1.0,
                               max_len=max_len)
    same = True
    for n, p in net.peft_model.named_parameters():
        if n in snap_before.get("lora_A", {}):
            same = same and bool(torch.allclose(p.data, snap_before["lora_A"][n]))
        elif n in snap_before.get("lora_B", {}):
            same = same and bool(torch.allclose(p.data, snap_before["lora_B"][n]))
    ok = out["logits_diff"] > 1e-3 and same
    RESULTS["test_modulation_p5a"] = {
        "logits_diff": out["logits_diff"], "weights_restored": same, "ok": ok}
    print(f"[P5-A] modulation logits diff={out['logits_diff']:.6f}  "
          f"weights_restored={same}  -> {ok}")
    assert ok, "P5-A modulation failed"
    return True


def run_all():
    base, net, texts_a, texts_b, cfg = make_fixture(max_len=96)
    tokenizer = base.tokenizer
    test_delta_existence(net, texts_a, tokenizer, cfg.max_len)
    test_state_dependency(net, texts_a, texts_b, tokenizer, cfg.max_len)
    test_parameter_transition(net, texts_a, tokenizer, cfg.max_len)
    test_behavioral_change(net, texts_a, tokenizer, cfg.max_len)
    test_modification_loop(net, texts_a, tokenizer, cfg.max_len)
    test_stability(net, texts_a, tokenizer, cfg.max_len, steps=20)
    test_modulation_p5a(net, texts_a, tokenizer, cfg.max_len)
    all_ok = all(v["ok"] for v in RESULTS.values())
    print("=== TESTS 1-6 + P5-A:", "PASS" if all_ok else "FAIL", "===")
    return RESULTS, all_ok


if __name__ == "__main__":
    import json

    results, ok = run_all()
    out_dir = os.path.join("experiments", "phase5")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "core_tests.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    sys.exit(0 if ok else 1)
