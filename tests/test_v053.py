"""Regression tests for v0.5.3 modules."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest


class TestExperienceCredit:
    def test_credit_computation(self):
        from dscns.experience_credit import ExperienceCreditAssigner
        a = ExperienceCreditAssigner(gamma=0.95, default_k=3)
        a.record_reward(1, 0.5)
        a.record_reward(2, -0.3)
        a.record_reward(3, 0.8)
        credit = a.compute_credit(1, k=3)
        expected = 0.5 + 0.95*(-0.3) + 0.95**2*0.8
        assert abs(credit.discounted_return - expected) < 1e-4

    def test_credit_windows(self):
        from dscns.experience_credit import ExperienceCreditAssigner
        a = ExperienceCreditAssigner(gamma=0.9)
        for i in range(1, 11):
            a.record_reward(i, float(i) / 10)
        credits = a.compute_multi_window_credits(1, windows=[1, 3, 5, 10])
        assert credits[1].discounted_return == credits[1].immediate_reward
        assert credits[10].credit_window == 10

    def test_credit_statistics(self):
        from dscns.experience_credit import ExperienceCreditAssigner
        a = ExperienceCreditAssigner()
        for i in range(1, 6):
            a.record_reward(i, 0.5 if i % 2 == 0 else -0.5)
            a.compute_credit(i)
        stats = a.credit_statistics()
        assert stats["n_credits"] == 5

    def test_credit_tracker(self):
        from dscns.experience_credit import ExperienceCreditAssigner, TemporalCreditTracker
        a = ExperienceCreditAssigner()
        t = TemporalCreditTracker()
        a.record_reward(1, 0.5)
        c = a.compute_credit(1)
        t.record(1, c, outcome="success")
        assert len(t.credit_history) == 1
        by = t.credit_by_outcome()
        assert "success" in by


class TestExperienceValue:
    def test_value_computation(self):
        from dscns.experience_value import ExperienceValueModel
        m = ExperienceValueModel()
        ev = m.record("e1", 1, reward=0.5, experience_type="success")
        assert ev.value > 0

    def test_verify(self):
        from dscns.experience_value import ExperienceValueModel
        m = ExperienceValueModel()
        m.record("e1", 1, reward=-0.3, experience_type="failure")
        m.verify("e1", success=True)
        assert m.values["e1"].n_successes == 1

    def test_ranking(self):
        from dscns.experience_value import ExperienceValueModel
        m = ExperienceValueModel()
        m.record("e1", 1, reward=0.8, experience_type="success")
        m.record("e2", 2, reward=-0.5, experience_type="failure")
        ranked = m.rank_by_value(["e1", "e2"])
        assert ranked[0] == "e1"

    def test_capacity(self):
        from dscns.experience_value import ExperienceValueModel
        m = ExperienceValueModel(capacity=10)
        for i in range(20):
            m.record(f"e{i}", i, reward=float(i))
        assert len(m.values) <= 10


class TestPolicyAdapter:
    def test_forward_shapes(self):
        from dscns.policy_adapter import PolicyAdapter
        pa = PolicyAdapter(state_dim=256, error_dim=32, memory_dim=32, value_dim=16,
                           hidden_dim=128, n_candidates=4)
        out = pa(torch.randn(2, 256), torch.randn(2, 32),
                 torch.randn(2, 32), torch.randn(2, 16))
        assert out["target_probs"].shape == (2, 3)
        assert out["magnitude"].shape == (2, 1)
        assert out["candidate_scores"].shape == (2, 4)

    def test_target_selection(self):
        from dscns.policy_adapter import PolicyAdapter
        pa = PolicyAdapter()
        probs = torch.tensor([[0.1, 0.7, 0.2]])
        sel = pa.select_target(probs, exploration_eps=0.0)
        assert sel.item() == 1

    def test_magnitude_range(self):
        from dscns.policy_adapter import PolicyAdapter
        pa = PolicyAdapter(m_min=0.02, m_max=1.0)
        out = pa(torch.randn(4, 256), torch.randn(4, 32),
                 torch.randn(4, 32), torch.randn(4, 16))
        assert out["magnitude"].min() >= 0.02
        assert out["magnitude"].max() <= 1.0


class TestPolicyLearning:
    def test_contrastive_loss(self):
        from dscns.policy_learning import ContrastiveExperienceLoss
        cl = ContrastiveExperienceLoss(margin=0.1)
        loss = cl(torch.tensor([0.8]), torch.tensor([0.3]), torch.tensor([0.9]))
        assert loss.item() >= 0
        # zero loss when well-separated
        loss_zero = cl(torch.tensor([0.9]), torch.tensor([0.1]), torch.tensor([0.9]))
        assert loss_zero.item() == 0.0

    def test_combined_loss(self):
        from dscns.policy_learning import ModificationPolicyLearner
        learner = ModificationPolicyLearner()
        new_p = torch.tensor([[0.5, 0.3, 0.2]])
        losses = learner.compute_loss(
            new_p, "failure", 0,
            failed_targets=torch.tensor([0]),
            success_targets=torch.tensor([2]))
        assert losses["total"].item() >= 0


class TestAlternativeProposal:
    def test_generation(self):
        from dscns.alternative_proposal import AlternativeProposalGenerator
        gen = AlternativeProposalGenerator(n_candidates=4)
        cands = gen.generate_candidates(base_target=0, base_magnitude=0.3)
        assert len(cands) == 4

    def test_selection(self):
        from dscns.alternative_proposal import AlternativeProposalGenerator
        gen = AlternativeProposalGenerator(n_candidates=4)
        cands = gen.generate_candidates(
            base_target=0, base_magnitude=0.3,
            candidate_scores=torch.tensor([[1.0, 0.5, 0.3, 0.1]]))
        sel = gen.select_candidate(cands, exploration_eps=0.0)
        assert sel.candidate_id == 0

    def test_data_leakage_prevention(self):
        from dscns.alternative_proposal import AlternativeProposalGenerator
        gen = AlternativeProposalGenerator(n_candidates=4)
        cands = gen.generate_candidates(base_target=0, base_magnitude=0.3)
        sel = gen.select_candidate(cands, exploration_eps=0.0)
        gen.record_outcome(sel, "success")
        # unexecuted candidates should NOT have an outcome
        for c in cands:
            if not c.selected:
                assert c.outcome == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
