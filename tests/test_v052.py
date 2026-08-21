"""v0.5.2 regression tests — Outcome-Conditioned Error-Driven Self-Modification.

Covers the v0.5.2 specification sections:

  §36  SelfModification       magnitude/target heads, apply_self_modification,
                              runtime parameter change
  §37  NoTopologyChange       parameter shapes / layers / modules / LoRA rank
                              preserved across 10 modification rounds
  §20  MemoryEncoder          direction + outcome embeddings, memory attention
  §9-10 WeightLearning        bounded weight, ranking loss, outcome-conditioned
                              modulation
  §5   ReplayWithMemory       direction + outcome in replay samples,
                              failure-weighted sampling
  §31  ExperienceAbsorption   experience tracker, failure->success lineage, EAR
  §32  FutureBehavior         future-evaluator metrics, success reuse, direction
                              similarity

Most tests drive the real modules (`intrinsic_plasticity.py`,
`networks.py`, `memory_encoder.py`, `experience_replay.py`) plus the new
`dscns/weight_learning.py`.  The v0.5.2-spec components that are not yet
part of the `dscns/` package — the §20 direction embedding, the §31
experience tracker / EAR and the §32 future-behavior evaluator — are
exercised through the small self-contained test doubles defined at the
bottom of this file; swap them for the real imports when those land.

Run:  python -m pytest tests/test_v052.py -v
"""
import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dscns.weight_learning import (
    WeightLearner, WeightRankingLoss, OutcomeConditionedWeight,
    W_MIN, W_MAX, DEFAULT_MARGIN,
)
from dscns.intrinsic_plasticity import IntrinsicPlasticityModule, NUM_TARGET_GROUPS
from dscns.networks import CognitiveNetwork
from dscns.memory import MemorySystem
from dscns.memory_encoder import MemoryEncoder, MemoryPolicyEncoder, ModificationEpisode
from dscns.experience_replay import ExperienceReplayBuffer, ReplayEntry


# ---------------------------------------------------------------------- #
# shared fixtures                                                        #
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def plasticity(device):
    """Real P5.1 plasticity module with magnitude + target heads (p51=True)."""
    torch.manual_seed(0)
    return IntrinsicPlasticityModule(
        hidden_dim=768, adapter_dim=16, meta_dim=32, plasticity_rank=8,
        p51=True, m_min=0.02, m_max=1.0, error_dim=32,
        num_target_groups=3,
    ).to(device).eval()


@pytest.fixture(scope="module")
def proposal_inputs(plasticity, device):
    """Random tensors shaped like one modification round's internal state."""
    B, T, H = 2, 8, 768
    hidden = torch.randn(B, T, H, device=device)
    current_params = {
        "W_A": torch.randn(H, 16, device=device) * 0.05,
        "W_B": torch.randn(16, H, device=device) * 0.05,
    }
    meta = torch.randn(B, 32, device=device)
    return hidden, current_params, meta


class _MockPeftModel(torch.nn.Module):
    """Minimal LoRA parameter stand-in mirroring the GPT-2 adapter layout.

    Parameter names follow the real PeftModel scheme so that
    CognitiveNetwork.apply_self_modification touches exactly the same
    (lora_A / lora_B) shapes it touches in production:

      attn c_attn  lora_A (16, 768),  lora_B (768, 16)
      attn c_proj  lora_A (16, 768),  lora_B (768, 16)
      mlp  c_proj  lora_A (16, 3072), lora_B (768, 16)   # A skipped (input 3072)
    """

    def __init__(self, adapter_id="N1", n_layers=2, rank=16,
                 hidden=768, mlp_in=3072):
        super().__init__()
        self.adapter_id = adapter_id
        self.device = torch.device("cpu")
        # plain dict: ParameterDict rejects "." in names, and these names
        # must stay dotted to match the real PeftModel naming scheme
        self._params = {}
        torch.manual_seed(7)
        for layer in range(n_layers):
            base = f"base_model.model.transformer.h.{layer}"
            for proj, in_dim in (("attn.c_attn", hidden),
                                 ("attn.c_proj", hidden),
                                 ("mlp.c_proj", mlp_in)):
                self._params[f"{base}.{proj}.lora_A.{adapter_id}.weight"] = \
                    torch.nn.Parameter(torch.randn(rank, in_dim) * 0.02)
                self._params[f"{base}.{proj}.lora_B.{adapter_id}.weight"] = \
                    torch.nn.Parameter(torch.randn(hidden, rank) * 0.02)

    def named_parameters(self, prefix="", recurse=True):
        for name, p in self._params.items():
            yield name, p


@pytest.fixture(scope="module")
def mock_net():
    """Real CognitiveNetwork wrapped around a lightweight LoRA stand-in."""
    model = _MockPeftModel(adapter_id="N1", n_layers=2, rank=16)
    return CognitiveNetwork(
        net_id="N1", name="WorldKnowledge", domain="general",
        peft_model=model, memory=MemorySystem(), plasticity=None,
    )


# ====================================================================== #
# §36 SelfModification (critical)                                        #
# ====================================================================== #
class TestSelfModification:
    """§36 — the learned self-modification primitives must be sound."""

    def test_magnitude_head_produces_nonzero(self, plasticity, proposal_inputs):
        hidden, params, meta = proposal_inputs
        proposal = plasticity.generate_proposal(hidden, params, meta)
        # full proposal path: magnitude is strictly inside (m_min, m_max)
        assert proposal["magnitude"] > 0.0
        assert proposal["magnitude"] >= 0.02 - 1e-9
        assert proposal["magnitude"] <= 1.0 + 1e-9
        # direct head check: sigmoid output is always positive
        extended_z = torch.randn(64, 256 + 2 * 32)
        with torch.no_grad():
            m_raw = torch.sigmoid(plasticity.magnitude_head(extended_z))
        assert bool((m_raw > 0.0).all())

    def test_target_head_produces_valid_target(self, plasticity, proposal_inputs):
        hidden, params, meta = proposal_inputs
        proposal = plasticity.generate_proposal(hidden, params, meta)
        assert proposal["target_group"] in (0, 1, 2)
        assert proposal["target_group"] in set(range(NUM_TARGET_GROUPS))
        # direct head check across many samples
        extended_z = torch.randn(64, 256 + 2 * 32)
        with torch.no_grad():
            probs = torch.softmax(plasticity.target_head(extended_z), dim=-1)
            targets = probs.argmax(dim=-1)
        assert bool((targets >= 0).all() and (targets < NUM_TARGET_GROUPS).all())

    def test_parameter_delta_nonzero(self, plasticity, proposal_inputs, mock_net):
        hidden, params, meta = proposal_inputs
        proposal = plasticity.generate_proposal(hidden, params, meta)
        # the generated delta itself is nonzero
        assert float(proposal["delta_W_A"].norm()) > 0.0
        assert float(proposal["delta_W_B"].norm()) > 0.0
        # applying it must move at least one parameter
        before = {n: p.detach().clone()
                  for n, p in mock_net.peft_model.named_parameters()}
        proposal["magnitude"] = 1.0  # force max magnitude for a robust check
        mock_net.apply_self_modification(proposal, alpha=1.0)
        total_change = 0.0
        for n, p in mock_net.peft_model.named_parameters():
            total_change += float((p.data - before[n]).abs().sum())
        assert total_change > 1e-8

    def test_runtime_parameter_change(self, plasticity, proposal_inputs, mock_net):
        hidden, params, meta = proposal_inputs
        theta_old = {n: p.detach().clone()
                     for n, p in mock_net.peft_model.named_parameters()}
        proposal = plasticity.generate_proposal(hidden, params, meta)
        mock_net.apply_self_modification(proposal, alpha=1.0)
        theta_new = {n: p.detach().clone()
                     for n, p in mock_net.peft_model.named_parameters()}
        changed = sum(1 for n in theta_old
                      if not torch.equal(theta_old[n], theta_new[n]))
        assert changed > 0


# ====================================================================== #
# §37 NoTopologyChange (critical)                                        #
# ====================================================================== #
class TestNoTopologyChange:
    """§37 — self-modification must never change the network topology."""

    @staticmethod
    def _shape_map(net):
        return {n: tuple(p.shape)
                for n, p in net.peft_model.named_parameters()}

    @staticmethod
    def _layer_ids(net):
        layers = set()
        for n in dict(net.peft_model.named_parameters()):
            if ".h." in n:
                layers.add(n.split(".h.")[1].split(".")[0])
        return layers

    @staticmethod
    def _module_count(net):
        return sum(1 for n in dict(net.peft_model.named_parameters())
                   if "lora" in n)

    @staticmethod
    def _lora_ranks(net):
        return {p.shape[0]
                for n, p in net.peft_model.named_parameters()
                if "lora_A" in n}

    def _apply_rounds(self, mock_net, plasticity, proposal_inputs, n_rounds=10):
        hidden, params, meta = proposal_inputs
        for i in range(n_rounds):
            proposal = plasticity.generate_proposal(hidden + i * 0.01,
                                                    params, meta)
            proposal["magnitude"] = 0.5
            mock_net.apply_self_modification(proposal, alpha=0.1)

    def test_parameter_shape_preserved(self, mock_net, plasticity,
                                       proposal_inputs):
        shapes_before = self._shape_map(mock_net)
        self._apply_rounds(mock_net, plasticity, proposal_inputs, n_rounds=10)
        shapes_after = self._shape_map(mock_net)
        assert set(shapes_before) == set(shapes_after)
        for name in shapes_before:
            assert shapes_before[name] == shapes_after[name], name

    def test_layer_count_unchanged(self, mock_net, plasticity,
                                   proposal_inputs):
        before = len(self._layer_ids(mock_net))
        self._apply_rounds(mock_net, plasticity, proposal_inputs, n_rounds=10)
        after = len(self._layer_ids(mock_net))
        assert before == 2
        assert after == before

    def test_module_count_unchanged(self, mock_net, plasticity,
                                    proposal_inputs):
        before = self._module_count(mock_net)
        self._apply_rounds(mock_net, plasticity, proposal_inputs, n_rounds=10)
        after = self._module_count(mock_net)
        assert before == 12            # 6 LoRA modules x 2 layers
        assert after == before

    def test_lora_rank_unchanged(self, mock_net, plasticity, proposal_inputs):
        before = self._lora_ranks(mock_net)
        self._apply_rounds(mock_net, plasticity, proposal_inputs, n_rounds=10)
        after = self._lora_ranks(mock_net)
        assert before == {16}          # LoRA rank stays 16
        assert after == before


# ====================================================================== #
# §20 MemoryEncoder                                                      #
# ====================================================================== #
class TestMemoryEncoder:
    """§20 — direction/outcome memory encodings and outcome-aware attention."""

    def test_direction_embedding_produces_vector(self, device):
        enc = _DirectionEncoder(flatten_dim=768 * 16 + 16 * 768,
                                out_dim=32).to(device)
        delta = {"delta_W_A": torch.randn(768, 16, device=device),
                 "delta_W_B": torch.randn(16, 768, device=device)}
        z = enc(delta)
        assert tuple(z.shape) == (32,)
        assert bool(torch.isfinite(z).all())
        assert float(z.norm()) > 0.0

    def test_direction_similarity_same_direction(self, device):
        enc = _DirectionEncoder(flatten_dim=768 * 16 + 16 * 768,
                                out_dim=32).to(device)
        delta = {"delta_W_A": torch.randn(768, 16, device=device),
                 "delta_W_B": torch.randn(16, 768, device=device)}
        z1 = enc(delta)
        z2 = enc(delta)                       # deterministic -> identical
        sim = float(torch.nn.functional.cosine_similarity(
            z1.unsqueeze(0), z2.unsqueeze(0)))
        assert sim > 0.99

    def test_direction_similarity_opposite(self, device):
        enc = _DirectionEncoder(flatten_dim=768 * 16 + 16 * 768,
                                out_dim=32).to(device)
        delta = {"delta_W_A": torch.randn(768, 16, device=device),
                 "delta_W_B": torch.randn(16, 768, device=device)}
        opposite = {"delta_W_A": -delta["delta_W_A"],
                    "delta_W_B": -delta["delta_W_B"]}
        z1 = enc(delta)
        z2 = enc(opposite)
        sim = float(torch.nn.functional.cosine_similarity(
            z1.unsqueeze(0), z2.unsqueeze(0)))
        assert sim < -0.99

    def test_outcome_embedding_success_failure_different(self, device):
        torch.manual_seed(0)
        ep_s = ModificationEpisode(category="success", outcome="success",
                                   proposal_norm=0.5, weight=0.8, target=0,
                                   delta_score=0.05, correction_norm=0.0)
        ep_f = ModificationEpisode(category="failure", outcome="failure",
                                   proposal_norm=0.5, weight=0.8, target=0,
                                   delta_score=-0.05, correction_norm=0.4)
        # size the encoder to the real episode feature vector (18 dims:
        # 2 proposal + 3 target + 1 delta + 8 error + 3 outcome + 1 correction)
        feat_dim = ep_s.to_feature_vector(device).shape[0]
        encoder = MemoryEncoder(feature_dim=feat_dim, memory_dim=32).to(device).eval()
        z_s = encoder.encode_single(ep_s, device)   # (1, 32)
        z_f = encoder.encode_single(ep_f, device)   # (1, 32)
        assert tuple(z_s.shape) == (1, 32)
        assert tuple(z_f.shape) == (1, 32)
        assert float((z_s - z_f).abs().sum()) > 1e-4

    def test_random_memory_not_zero(self, device):
        torch.manual_seed(0)
        encoder = MemoryEncoder().to(device).eval()
        z_rand = encoder(torch.randn(4, 15, device=device))
        z_zero = encoder(torch.zeros(4, 15, device=device))
        # L2-normalized output can never collapse to the zero vector
        assert bool(torch.any(z_rand != 0.0))
        # random memory is distinguishable from the zero memory
        assert float((z_rand - z_zero).abs().sum()) > 1e-4

    def test_zero_memory_all_zeros(self, device):
        # a zero delta carries no direction -> embedding is exactly zero
        enc = _DirectionEncoder(flatten_dim=768 * 16 + 16 * 768,
                                out_dim=32).to(device)
        delta = {"delta_W_A": torch.zeros(768, 16, device=device),
                 "delta_W_B": torch.zeros(16, 768, device=device)}
        z = enc(delta)
        assert bool((z == 0.0).all())

    def test_outcome_aware_attention(self, device):
        torch.manual_seed(0)
        encoder = MemoryPolicyEncoder(memory_dim=32, hidden_dim=64).to(device).eval()
        memory_z = torch.randn(8, 32, device=device)
        with torch.no_grad():
            z_summary, attn = encoder(memory_z)
        assert tuple(attn.shape) == (8,)
        assert abs(float(attn.sum()) - 1.0) < 1e-5
        assert tuple(z_summary.shape) == (32,)
        # batched variant: attention over the k memories sums to 1 per row
        memory_batch = torch.randn(3, 8, 32, device=device)
        with torch.no_grad():
            _, attn_b = encoder(memory_batch)
        assert tuple(attn_b.shape) == (3, 8)
        assert bool(torch.allclose(attn_b.sum(dim=1), torch.ones(3, device=device),
                                   atol=1e-5))


# ====================================================================== #
# §9-10 WeightLearning                                                   #
# ====================================================================== #
class TestWeightLearning:
    """§9-10 — learned modification weight w_t ∈ [w_min, w_max]."""

    @pytest.fixture(scope="class")
    def learner(self):
        torch.manual_seed(1)
        return WeightLearner(state_dim=256, error_dim=32, memory_dim=32,
                             hidden=64, w_min=W_MIN, w_max=W_MAX).eval()

    def test_weight_bounded(self, learner):
        for _ in range(5):
            w = float(learner(torch.randn(256), torch.randn(32),
                              torch.randn(32)))
            assert W_MIN - 1e-6 <= w <= W_MAX + 1e-6
        # batched forward keeps the same bounds
        wb = learner(torch.randn(16, 256), torch.randn(16, 32),
                     torch.randn(16, 32))
        assert tuple(wb.shape) == (16, 1)
        assert bool((wb >= W_MIN - 1e-6).all() and (wb <= W_MAX + 1e-6).all())

    def test_ranking_loss_positive_when_wrong(self, learner):
        # w_failure > w_success violates the ordering -> positive loss
        loss = learner.ranking_loss(w_success=0.2, w_failure=0.9,
                                    margin=DEFAULT_MARGIN)
        assert float(loss) > 0.0
        crit = WeightRankingLoss(margin=DEFAULT_MARGIN)
        assert float(crit(w_success=0.2, w_failure=0.9)) > 0.0
        assert float(crit.loss_online(0.2, 0.9)) > 0.0
        assert float(crit.loss_batch(torch.tensor([0.2, 0.4]),
                                     torch.tensor([0.9, 0.6]))) > 0.0

    def test_ranking_loss_zero_when_correct(self, learner):
        # w_success - w_failure >= margin -> loss is exactly zero
        loss = learner.ranking_loss(w_success=0.9, w_failure=0.2,
                                    margin=DEFAULT_MARGIN)
        assert float(loss) == 0.0
        crit = WeightRankingLoss(margin=DEFAULT_MARGIN)
        assert float(crit(w_success=0.9, w_failure=0.2)) == 0.0

    def test_weight_increases_after_success(self):
        w = OutcomeConditionedWeight(w_min=W_MIN, w_max=W_MAX)
        w1 = w.update_from_outcome(0.5, "success")
        assert w1 > 0.5
        assert W_MIN <= w1 <= W_MAX

    def test_weight_decreases_after_failure(self):
        w = OutcomeConditionedWeight(w_min=W_MIN, w_max=W_MAX)
        w1 = w.update_from_outcome(0.5, "failure")
        assert w1 < 0.5
        assert W_MIN <= w1 <= W_MAX

    def test_recovery_raises_correction_weight(self):
        w = OutcomeConditionedWeight(w_min=W_MIN, w_max=W_MAX)
        assert w.w_correction == 0.0
        w.update_from_outcome(0.4, "recovery")
        assert w.w_correction > 0.0

    def test_repeated_similar_failure_penalizes_more(self):
        w = OutcomeConditionedWeight(w_min=W_MIN, w_max=W_MAX)
        w_first = w.update_from_outcome(0.5, "failure", similarity_to_past=0.0)
        w_repeat = w.update_from_outcome(0.5, "failure", similarity_to_past=0.9)
        assert w_repeat < w_first
        assert w_repeat < 0.5 - 0.1 - 0.1 + 1e-9  # failure + repeat penalty


# ====================================================================== #
# §5 ReplayWithMemory                                                    #
# ====================================================================== #
class TestReplayWithMemory:
    """§5 — replay samples carry modification direction and outcome."""

    @pytest.fixture(scope="class")
    def buffer(self):
        buf = ExperienceReplayBuffer(capacity=200, failure_ratio=1.0)
        for i in range(40):
            buf.add(ReplayEntry(
                core_z=torch.randn(256),
                prev_delta_A=torch.randn(768, 16),
                prev_delta_B=torch.randn(16, 768),
                prev_weight=0.5, prev_target=i % 3,
                delta_score=0.1, outcome="success", category="success",
                reward=0.5, round_id=i,
            ))
        for i in range(10):
            buf.add(ReplayEntry(
                core_z=torch.randn(256),
                prev_delta_A=torch.randn(768, 16),
                prev_delta_B=torch.randn(16, 768),
                prev_weight=0.2, prev_target=i % 3,
                delta_score=-0.2, outcome="failure", category="failure",
                reward=-0.5, round_id=100 + i,
            ))
        return buf

    def test_replay_sample_includes_direction(self, buffer):
        sample = buffer.sample(8, strategy="uniform")
        assert len(sample) == 8
        for s in sample:
            assert "prev_delta_A" in s and "prev_delta_B" in s
            assert tuple(s["prev_delta_A"].shape) == (768, 16)
            assert tuple(s["prev_delta_B"].shape) == (16, 768)
            direction = torch.cat([torch.flatten(s["prev_delta_A"]),
                                   torch.flatten(s["prev_delta_B"])])
            assert float(direction.norm()) > 0.0   # a real delta_theta direction

    def test_replay_sample_includes_outcome(self, buffer):
        # the buffer keeps the outcome label with every stored experience
        assert buffer.success_count == 40
        assert buffer.failure_count == 10
        assert all(e.category == "success" and e.outcome == "success"
                   for e in buffer.get_success_entries())
        assert all(e.category == "failure" and e.outcome == "failure"
                   for e in buffer.get_failure_entries())
        # training pairs expose the outcome signal (delta_score / reward)
        pair = buffer.sample(4, strategy="uniform")[0]
        assert "delta_score" in pair and "reward" in pair
        assert "prev_weight" in pair and "prev_target" in pair

    def test_failure_weighted_sampling(self, buffer):
        import random
        random.seed(0)
        batch_size = 20
        indices = buffer._failure_weighted_sample(batch_size)
        sampled_failures = sum(1 for i in indices
                               if buffer.entries[i].category == "failure")
        natural_rate = buffer.failure_count / buffer.total_count  # 10/50 = 0.2
        sampled_rate = sampled_failures / len(indices)
        # with failure_ratio=1.0 the draw is exactly half failures
        assert sampled_rate == 0.5
        assert sampled_rate > natural_rate + 0.2


# ====================================================================== #
# §31 ExperienceAbsorption                                               #
# ====================================================================== #
class TestExperienceAbsorption:
    """§31 — experience recording, failure->success lineage, EAR."""

    def test_experience_tracker_records(self):
        tracker = ExperienceTracker()
        assert len(tracker.experiences) == 0
        tracker.record({"round_id": 0, "category": "failure", "delta_score": -0.2})
        tracker.record({"round_id": 1, "category": "success", "delta_score": 0.1})
        tracker.record({"round_id": 2, "category": "failure", "delta_score": -0.1})
        assert len(tracker.experiences) == 3
        assert tracker.experiences[1]["category"] == "success"
        assert tracker.experiences[2]["delta_score"] == -0.1

    def test_lineage_links_failure_to_success(self):
        tracker = ExperienceTracker()
        tracker.record({"round_id": 5, "category": "failure"})
        tracker.record({"round_id": 6, "category": "failure"})
        tracker.record({"round_id": 7, "category": "success"})
        # the most recent failure is linked to the following success
        assert tracker.lineage == [(6, 7)]

    def test_ear_positive_when_absorption(self):
        absorb = ExperienceAbsorption(initial_rfr=1.0)
        # before absorption RFR = 1.0 (failures always repeat); after
        # absorbing the experience the repeat-failure rate drops -> EAR > 0
        ear1 = absorb.absorb({"category": "failure"}, rfr_after=0.5)
        ear2 = absorb.absorb({"category": "failure"}, rfr_after=0.4)
        assert ear1 > 0.0 and ear2 > 0.0
        assert absorb.EAR > 0.0
        assert absorb.EAR == pytest.approx((0.5 + 0.1) / 2.0)


# ====================================================================== #
# §32 FutureBehavior                                                     #
# ====================================================================== #
class TestFutureBehavior:
    """§32 — future-behavior evaluation: metrics, reuse, similarity."""

    def test_future_evaluator_computes_all_metrics(self):
        ev = FutureBehaviorEvaluator()
        d1 = {"delta_W_A": torch.randn(768, 16), "delta_W_B": torch.randn(16, 768)}
        d2 = {"delta_W_A": torch.randn(768, 16), "delta_W_B": torch.randn(16, 768)}
        ev.record_modification(d1, reused=True, previous_delta=d1)
        ev.record_modification(d2, reused=True, previous_delta=d1)
        ev.record_modification(d1, reused=False, previous_delta=d2)
        out = ev.evaluate()
        for key in ("success_reuse_rate", "modification_similarity",
                    "total_modifications", "reused_modifications"):
            assert key in out
        assert out["total_modifications"] == 3
        assert out["reused_modifications"] == 2
        assert out["success_reuse_rate"] == pytest.approx(2 / 3)
        assert -1.0 <= out["modification_similarity"] <= 1.0

    def test_success_reuse_tracking(self):
        ev = FutureBehaviorEvaluator()
        ev.record_modification({"delta_W_A": torch.zeros(1),
                                "delta_W_B": torch.zeros(1)}, reused=True)
        ev.record_modification({"delta_W_A": torch.zeros(1),
                                "delta_W_B": torch.zeros(1)}, reused=False)
        assert ev.reused_modifications == 1
        assert ev.total_modifications == 2
        assert ev.evaluate()["success_reuse_rate"] == 0.5

    def test_modification_similarity(self):
        ev = FutureBehaviorEvaluator()
        d = {"delta_W_A": torch.randn(768, 16), "delta_W_B": torch.randn(16, 768)}
        same = {"delta_W_A": d["delta_W_A"].clone(), "delta_W_B": d["delta_W_B"].clone()}
        opposite = {"delta_W_A": -d["delta_W_A"], "delta_W_B": -d["delta_W_B"]}
        assert ev.direction_similarity(d, same) > 0.99
        assert ev.direction_similarity(d, opposite) < -0.99


# ====================================================================== #
# v0.5.2 test doubles (components not yet part of the dscns/ package)    #
# ====================================================================== #
class _DirectionEncoder(torch.nn.Module):
    """v0.5.2 §20 test double: delta_theta -> unit direction embedding.

    A bias-free linear projection followed by L2 normalization: the
    embedding of ``delta`` is the normalized *direction* of the
    modification, and the zero delta maps to the exact zero vector
    (a modification with no direction).
    """

    def __init__(self, flatten_dim=768 * 16 + 16 * 768, out_dim=32, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.proj = torch.nn.Linear(flatten_dim, out_dim, bias=False)

    def forward(self, delta):
        a = torch.flatten(torch.as_tensor(delta["delta_W_A"], dtype=torch.float32))
        b = torch.flatten(torch.as_tensor(delta["delta_W_B"], dtype=torch.float32))
        v = torch.cat([a, b])
        z = self.proj(v)
        n = z.norm()
        if float(n) < self.eps:
            return torch.zeros_like(z)
        return z / n


class ExperienceTracker:
    """v0.5.2 §31 test double: records experiences and failure->success lineage.

    ``lineage`` stores (failure_round, success_round) pairs: a success that
    immediately follows a recorded failure is linked to it, which is the
    signal that the error experience was absorbed.
    """

    def __init__(self):
        self.experiences = []
        self.lineage = []
        self._last_failure = None

    def record(self, experience):
        record = dict(experience)
        round_id = record.get("round_id", len(self.experiences))
        category = record.get("category", record.get("outcome", "neutral"))
        self.experiences.append(record)
        if category == "failure":
            self._last_failure = round_id
        elif category in ("success", "recovery") and self._last_failure is not None:
            self.lineage.append((self._last_failure, round_id))
            self._last_failure = None
        return len(self.experiences)


class ExperienceAbsorption:
    """v0.5.2 §31 test double: Experience Absorption Rate (EAR).

    EAR measures how much the Repeat Failure Rate (RFR) drops once an
    experience is absorbed:

        EAR = RFR_before - RFR_after

    A positive EAR means future behavior changed — the failure did not
    repeat with the same conditions.
    """

    def __init__(self, initial_rfr=1.0):
        self.rfr = float(initial_rfr)
        self.ears = []

    def absorb(self, experience, rfr_after):
        ear = self.rfr - float(rfr_after)
        self.ears.append(float(ear))
        self.rfr = float(rfr_after)
        return float(ear)

    @property
    def EAR(self):
        return float(np.mean(self.ears)) if self.ears else 0.0


class FutureBehaviorEvaluator:
    """v0.5.2 §32 test double: evaluate future modification behavior.

    Metrics:
      - success_reuse_rate     : fraction of modifications that reuse a
                                 previously successful modification
      - modification_similarity: mean cosine similarity between successive
                                 modification directions
    """

    def __init__(self):
        self.reused_modifications = 0
        self.total_modifications = 0
        self.direction_similarities = []

    def record_modification(self, delta, reused=False, previous_delta=None):
        self.total_modifications += 1
        if reused:
            self.reused_modifications += 1
        if previous_delta is not None:
            self.direction_similarities.append(
                self.direction_similarity(previous_delta, delta))

    def direction_similarity(self, d1, d2):
        a = torch.flatten(torch.as_tensor(d1["delta_W_A"], dtype=torch.float32))
        b = torch.flatten(torch.as_tensor(d2["delta_W_A"], dtype=torch.float32))
        c = torch.flatten(torch.as_tensor(d1["delta_W_B"], dtype=torch.float32))
        d = torch.flatten(torch.as_tensor(d2["delta_W_B"], dtype=torch.float32))
        v1 = torch.cat([a, c])
        v2 = torch.cat([b, d])
        n1, n2 = v1.norm(), v2.norm()
        if float(n1) < 1e-12 or float(n2) < 1e-12:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)))

    def evaluate(self):
        return {
            "success_reuse_rate": self.reused_modifications / max(1, self.total_modifications),
            "modification_similarity": float(np.mean(self.direction_similarities))
                if self.direction_similarities else 0.0,
            "total_modifications": self.total_modifications,
            "reused_modifications": self.reused_modifications,
        }
