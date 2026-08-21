"""Smoke test for v0.5.3 modules.

Verifies all new modules can be instantiated and run basic operations
without errors.  No GPU required.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_experience_credit():
    """Test temporal credit assignment."""
    from dscns.experience_credit import ExperienceCreditAssigner, TemporalCreditTracker

    assigner = ExperienceCreditAssigner(gamma=0.95, default_k=3)

    # record some rewards
    assigner.record_reward(1, 0.5, {"target": 0})
    assigner.record_reward(2, -0.3, {"target": 1})
    assigner.record_reward(3, 0.8, {"target": 2})
    assigner.record_reward(4, 0.2, {"target": 0})
    assigner.record_reward(5, -0.1, {"target": 1})

    # compute credit
    credit = assigner.compute_credit(1, k=3)
    assert credit.round_id == 1
    assert credit.credit_window == 3
    # G_1 = 0.5 + 0.95*(-0.3) + 0.95^2*(0.8) = 0.5 - 0.285 + 0.722 = 0.937
    expected = 0.5 + 0.95 * (-0.3) + (0.95**2) * 0.8
    assert abs(credit.discounted_return - expected) < 1e-4, f"Expected {expected}, got {credit.discounted_return}"

    # multi-window credits
    credits = assigner.compute_multi_window_credits(1, windows=[1, 3, 5])
    assert 1 in credits and 3 in credits and 5 in credits
    assert credits[1].discounted_return == 0.5  # immediate only

    # statistics
    stats = assigner.credit_statistics()
    assert stats["n_credits"] > 0

    # tracker
    tracker = TemporalCreditTracker()
    tracker.record(1, credit, {"target": 0}, "success")
    assert len(tracker.credit_history) == 1
    trajectory = tracker.credit_trajectory()
    assert len(trajectory) == 1
    by_outcome = tracker.credit_by_outcome()
    assert "success" in by_outcome

    print("  [PASS] experience_credit")


def test_experience_value():
    """Test experience value model."""
    from dscns.experience_value import ExperienceValueModel

    model = ExperienceValueModel(learning_rate=0.1, capacity=100)

    # record experiences
    ev1 = model.record("exp-1", 10, reward=0.5, experience_type="success",
                       target_group=0, magnitude=0.3)
    ev2 = model.record("exp-2", 15, reward=-0.3, experience_type="failure",
                       target_group=1, magnitude=0.8)

    assert ev1.value > 0  # success has positive value
    assert ev2.value < 0  # failure has negative value

    # verify
    model.verify("exp-2", success=True)
    ev2_updated = model.values["exp-2"]
    assert ev2_updated.n_verifications == 1
    assert ev2_updated.n_successes == 1

    # rank
    ranked = model.rank_by_value(["exp-1", "exp-2"])
    assert ranked[0] == "exp-1"  # success ranked higher

    # summary
    summary = model.summary()
    assert summary["n_experiences"] == 2

    # get failure/success values
    failures = model.get_failure_values()
    successes = model.get_success_values()
    assert len(failures) == 1
    assert len(successes) == 1

    print("  [PASS] experience_value")


def test_policy_adapter():
    """Test policy adapter."""
    from dscns.policy_adapter import PolicyAdapter

    adapter = PolicyAdapter(
        state_dim=256, error_dim=32, memory_dim=32, value_dim=16,
        hidden_dim=128, n_candidates=4, n_target_groups=3)

    B = 2
    state_z = torch.randn(B, 256)
    error_z = torch.randn(B, 32)
    memory_z = torch.randn(B, 32)
    value_z = torch.randn(B, 16)

    out = adapter(state_z, error_z, memory_z, value_z)

    assert out["target_logits"].shape == (B, 3)
    assert out["target_probs"].shape == (B, 3)
    assert out["magnitude"].shape == (B, 1)
    assert out["direction_mod"].shape == (B, 32)
    assert out["candidate_scores"].shape == (B, 4)
    assert out["confidence"].shape == (B, 1)

    # target probs should sum to ~1
    probs_sum = out["target_probs"].sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(B), atol=1e-4)

    # magnitude should be in [m_min, m_max]
    assert out["magnitude"].min() >= 0.02
    assert out["magnitude"].max() <= 1.0

    # select target
    selected = adapter.select_target(out["target_probs"], exploration_eps=0.0)
    assert selected.shape == (B,)
    assert all(0 <= s < 3 for s in selected.tolist())

    print("  [PASS] policy_adapter")


def test_policy_learning():
    """Test policy learning losses."""
    from dscns.policy_learning import (
        ModificationPolicyLearner, ContrastiveExperienceLoss,
        FailureAvoidanceLoss, SuccessReuseLoss, StabilityLoss)

    # contrastive loss
    cl = ContrastiveExperienceLoss(margin=0.1)
    s_score = torch.tensor([0.8])
    f_score = torch.tensor([0.3])
    sim = torch.tensor([0.9])
    loss_c = cl(s_score, f_score, sim)
    assert loss_c.item() >= 0

    # when success score is already higher than failure + margin, loss = 0
    s_high = torch.tensor([0.9])
    f_low = torch.tensor([0.1])
    loss_zero = cl(s_high, f_low, sim)
    assert loss_zero.item() == 0.0

    # avoid loss
    al = FailureAvoidanceLoss()
    target_probs = torch.tensor([[0.7, 0.2, 0.1]])
    failed_t = torch.tensor([0])  # target 0 failed
    loss_a = al(target_probs, failed_t)
    assert loss_a.item() > 0  # should penalize high prob on failed target

    # reuse loss
    rl = SuccessReuseLoss()
    success_t = torch.tensor([0])  # target 0 succeeded
    loss_r = rl(target_probs, success_t)
    assert loss_r.item() >= 0  # should reward high prob on success target

    # stability loss
    sl = StabilityLoss(max_kl=0.5)
    old_p = torch.tensor([[0.33, 0.33, 0.34]])
    new_p = torch.tensor([[0.6, 0.2, 0.2]])
    loss_s = sl(new_p, old_p)
    assert loss_s.item() >= 0

    # combined learner
    learner = ModificationPolicyLearner(lr=3e-4)
    new_probs = torch.tensor([[0.5, 0.3, 0.2]])
    losses = learner.compute_loss(
        new_probs, "failure", 0,
        failed_targets=torch.tensor([0]),
        success_targets=torch.tensor([2]))
    assert "total" in losses
    assert losses["total"].item() >= 0

    stats = learner.loss_statistics()
    assert stats["n_steps"] == 1

    print("  [PASS] policy_learning")


def test_alternative_proposal():
    """Test alternative proposal generation."""
    from dscns.alternative_proposal import AlternativeProposalGenerator, ModificationCandidate

    gen = AlternativeProposalGenerator(n_candidates=4, n_target_groups=3)

    # generate candidates
    candidates = gen.generate_candidates(
        base_target=0, base_magnitude=0.3,
        candidate_scores=torch.tensor([[1.0, 0.5, 0.3, 0.1]]))

    assert len(candidates) == 4
    assert candidates[0].target_group == 0  # base
    assert candidates[1].target_group == 1  # switched
    assert candidates[2].target_group == 0  # same target, different mag
    assert candidates[3].target_group == 2  # third group

    # select (exploitation: highest score)
    selected = gen.select_candidate(candidates, exploration_eps=0.0)
    assert selected.selected
    assert selected.candidate_id == 0  # highest score

    # record outcome
    gen.record_outcome(selected, "success")
    assert selected.outcome == "success"

    # unexecuted candidates should remain unknown
    assert candidates[1].outcome == "unknown"

    # stats
    summary = gen.summary()
    assert summary["n_selected"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["failure_rate"] == 0.0

    print("  [PASS] alternative_proposal")


def compute_policy_divergence(policy_A_dist, policy_B_dist):
    """Inlined from run_v053 for smoke test independence."""
    if not policy_A_dist or not policy_B_dist:
        return {"kl": 0.0, "js": 0.0, "cosine": 0.0, "n_samples": 0}
    n_targets = 3
    def _mean_dist(dists):
        counts = np.zeros(n_targets)
        for d in dists:
            for t, p in d.items():
                counts[min(int(t), n_targets-1)] += p
        counts /= max(len(dists), 1)
        total = counts.sum()
        if total > 0:
            counts /= total
        else:
            counts = np.ones(n_targets) / n_targets
        return counts
    p = _mean_dist(policy_A_dist)
    q = _mean_dist(policy_B_dist)
    eps = 1e-8
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    norm = np.linalg.norm(p) * np.linalg.norm(q)
    cosine = float(np.dot(p, q) / max(norm, 1e-12))
    return {"kl": kl, "js": js, "cosine": cosine, "n_samples": len(policy_A_dist)}


def test_policy_divergence():
    """Test policy divergence computation."""

    dist_A = [{0: 0.7, 1: 0.2, 2: 0.1}] * 10
    dist_B = [{0: 0.1, 1: 0.2, 2: 0.7}] * 10

    div = compute_policy_divergence(dist_A, dist_B)
    assert div["kl"] > 0, "KL should be > 0 for different distributions"
    assert div["js"] > 0, "JS should be > 0 for different distributions"
    assert div["cosine"] < 1.0, "Cosine should be < 1 for different distributions"

    # same distribution
    div_same = compute_policy_divergence(dist_A, dist_A)
    assert div_same["kl"] < 0.01, "KL should be ~0 for same distribution"
    assert div_same["js"] < 0.01, "JS should be ~0 for same distribution"

    print("  [PASS] policy_divergence")


def test_config():
    """Test v0.5.3 config fields exist."""
    from dscns.config import DSCNSConfig

    cfg = DSCNSConfig()
    assert hasattr(cfg, "v053_enabled")
    assert hasattr(cfg, "credit_gamma")
    assert hasattr(cfg, "credit_default_k")
    assert hasattr(cfg, "exp_value_learning_rate")
    assert hasattr(cfg, "policy_adapter_hidden")
    assert hasattr(cfg, "exploration_eps")
    assert hasattr(cfg, "lambda_contrastive")
    assert hasattr(cfg, "checkpoint_rounds")
    assert hasattr(cfg, "policy_diagnose_every")

    # from_dict
    d = {"v053_enabled": True, "credit_gamma": 0.9, "exploration_eps": 0.2}
    cfg2 = DSCNSConfig.from_dict(d)
    assert cfg2.v053_enabled is True
    assert cfg2.credit_gamma == 0.9

    print("  [PASS] config")


def main():
    print("\n" + "=" * 60)
    print("v0.5.3 Smoke Test")
    print("=" * 60)

    tests = [
        ("experience_credit", test_experience_credit),
        ("experience_value", test_experience_value),
        ("policy_adapter", test_policy_adapter),
        ("policy_learning", test_policy_learning),
        ("alternative_proposal", test_alternative_proposal),
        ("policy_divergence", test_policy_divergence),
        ("config", test_config),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
