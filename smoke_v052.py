# -*- coding: utf-8 -*-
"""Smoke test for v0.5.2 modules: experience_absorption, future_behavior,
and the new memory_encoder classes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dscns.experience_absorption import (
    ExperienceTracker, ExperienceLineage, AbsorptionEvaluator,
    SimilarityOutcomeTracker,
)
from dscns.future_behavior import (
    FutureModificationEvaluator, SuccessReuseEvaluator,
    ModificationSimilarityTracker,
)
from dscns.memory_encoder import (
    MemoryEncoder, ModificationEpisode, DirectionEncoder, OutcomeEmbedding,
    TrainedMemoryEncoder, OutcomeAwareAttention,
)

ok = []

# ---------------- 1. experience_absorption ----------------
tr = ExperienceTracker()
fid1 = tr.record_failure(round_id=1, target=1, magnitude=0.08,
                         delta_theta={"delta_W_A": np.zeros((768, 16))},
                         error=np.zeros(8),
                         weight_before=0.05, weight_after=0.02)
fid2 = tr.record_failure(round_id=3, target=2, magnitude=0.07)
mid = tr.record_modification(round_id=5, target=1, magnitude=0.03,
                             outcome="success", source_experience_ids=[fid1],
                             future_similarity=0.72)
assert fid1 != fid2 and fid1.startswith("exp-")
lin = tr.lineage_for(fid1)
assert lin is not None and lin.n_uses == 1 and lin.n_successes == 1
assert tr.absorption_rate() == 0.5, tr.absorption_rate()
print("ExperienceTracker lineage:", tr.summary())
ok.append("experience_absorption: ExperienceTracker + lineage")

# EAR normal case
ev = AbsorptionEvaluator()
r = ev.evaluate(rfr_baseline=0.4, rfr_future=0.1)
assert abs(r["ear"] - (1 - 0.1 / 0.4)) < 1e-9 and r["absorbed"]
print("EAR normal:", r)
ok.append("experience_absorption: AbsorptionEvaluator normal")

# EAR baseline=0, future=0 -> graceful
r0 = ev.evaluate(rfr_baseline=0.0, rfr_future=0.0)
assert r0["ear"] == 0.0 and r0["baseline_zero"] and not r0["absorbed"]
print("EAR baseline0/future0:", r0)
# EAR baseline=0, future>0 -> clamped -1
r1 = ev.evaluate(rfr_baseline=0.0, rfr_future=0.2)
assert r1["ear"] == -1.0 and r1["baseline_zero"]
print("EAR baseline0/future>0:", r1)
ok.append("experience_absorption: AbsorptionEvaluator baseline-0 safe")

# compute_rfr from tracker history
tr2 = ExperienceTracker()
for rt, tg, out in [(1, 0, "failure"), (2, 0, "failure"), (3, 1, "success"),
                    (4, 0, "failure"), (5, 0, "failure"), (6, 1, "success")]:
    tr2.record(round_id=rt, target=tg, outcome=out)
rfr_all = AbsorptionEvaluator.compute_rfr(tr2.experiences)
print("RFR all:", rfr_all)
res = ev.evaluate_from_tracker(tr2, baseline_window=3, future_window=3)
print("EAR from tracker:", res)
ok.append("experience_absorption: compute_rfr + evaluate_from_tracker")

# SimilarityOutcomeTracker
sot = SimilarityOutcomeTracker()
for rnd in range(8):
    sim = 0.9 if rnd < 4 else 0.3
    out = "failure" if (rnd < 4 and rnd % 2 == 0) else "success"
    sot.update(rnd, sim, out)
assert sot.high_similarity_success_rate(0.5) > 0
print("SimilarityOutcomeTracker:", sot.to_dict())
ok.append("experience_absorption: SimilarityOutcomeTracker")

# ---------------- 2. future_behavior ----------------
ev2 = FutureModificationEvaluator()
rng = np.random.RandomState(0)
def dtheta(seed):
    r = np.random.RandomState(seed)
    return {"delta_W_A": r.randn(768, 16), "delta_W_B": r.randn(16, 768)}
# historical: success at 1, failure at 2
ev2.record_experience(round_id=1, outcome="success", delta_theta=dtheta(1),
                      target=0, magnitude=0.05)
ev2.record_experience(round_id=2, outcome="failure", delta_theta=dtheta(2),
                      target=1, magnitude=0.09)
# future: reuse the success direction (should succeed), avoid failure dir
ev2.record_future_modification(round_id=3, delta_theta=dtheta(1), outcome="success",
                               source_experience_ids=["exp-x"])
ev2.record_future_modification(round_id=4, delta_theta=dtheta(99), outcome="success",
                               target=2, magnitude=0.02)
rep = ev2.evaluate()
print("FutureModificationEvaluator:", {k: v for k, v in rep.items()
                                       if k != "reuse_cells"})
assert "repeat_failure_rate" in rep and "successful_reuse_rate" in rep
assert "modification_similarity_shift" in rep
assert "target_shift" in rep and "magnitude_shift" in rep
assert rep["p_future_given_experience"]["explicit"] > 0
ok.append("future_behavior: FutureModificationEvaluator")

mst = ModificationSimilarityTracker()
for r in range(6):
    mst.add(dtheta(r % 2), round_id=r, outcome="success" if r % 2 == 0 else "failure",
            magnitude=0.05 + 0.01 * r, target=r % 3)
print("similarity_shift:", mst.similarity_shift(2))
print("target_shift:", mst.target_shift(2))
print("magnitude_shift:", mst.magnitude_shift(2))
sim = mst.similarity(dtheta(1), reference="success")
print("sim to success:", sim)
ok.append("future_behavior: ModificationSimilarityTracker")

sre = SuccessReuseEvaluator()
sre.record_success(1, dtheta(1), target=0)
sre.record_failure(2, dtheta(2), target=1)
sre.record_future_modification(3, dtheta(1), outcome="success", target=0)
sre.record_future_modification(4, dtheta(2), outcome="success", target=0)  # avoidance
sre.record_future_modification(5, dtheta(2), outcome="failure", target=1)  # repeat
print("SuccessReuseEvaluator:", sre.reuse_summary())
summ = sre.reuse_summary()
assert summ["reused_successes"] == 1 and summ["avoided_failures"] == 1
assert summ["repeated_failures"] == 1
ok.append("future_behavior: SuccessReuseEvaluator (reuse vs avoidance)")

# ---------------- 3. memory_encoder v0.5.2 ----------------
dA = torch.randn(768, 16)
dB = torch.randn(16, 768)
enc = DirectionEncoder()
zd = enc(dA, dB)
assert zd.shape == (1, 16)
sim = DirectionEncoder.cosine_similarity(zd, zd)
assert abs(float(sim) - 1.0) < 1e-5
# PCA fit path
samples = [(torch.randn(768, 16), torch.randn(16, 768)) for _ in range(20)]
enc.fit_pca(samples)
zd_pca = enc(torch.randn(768, 16), torch.randn(16, 768))
assert zd_pca.shape == (1, 16)
print("DirectionEncoder: learned + pca ok, cos_sim =", float(sim))
ok.append("memory_encoder: DirectionEncoder")

oe = OutcomeEmbedding()
e_s = oe.embed("success"); e_f = oe.embed("failure"); e_r = oe.embed("recovery")
assert e_s.shape == (8,) and e_f.shape == (8,) and e_r.shape == (8,)
assert not torch.allclose(e_s, e_f)
print("OutcomeEmbedding: success != failure vectors, trainable =",
      oe.embedding.weight.requires_grad)
ok.append("memory_encoder: OutcomeEmbedding")

base = MemoryEncoder()
tme = TrainedMemoryEncoder(base_encoder=base)
feat = torch.randn(3, 15)
core = torch.randn(3, 256)
out = tme(feat, core, dA.unsqueeze(0).expand(3, -1, -1),
          dB.unsqueeze(0).expand(3, -1, -1), outcome=["success", "failure", "recovery"])
assert out["z_memory"].shape == (3, 32)
assert out["z_direction"].shape == (3, 16)
assert out["z_outcome"].shape == (3, 8)
# trainability check: different memory conditions -> different z_memory
with torch.no_grad():
    z_a = tme(feat, core, dA.unsqueeze(0).expand(3, -1, -1),
              dB.unsqueeze(0).expand(3, -1, -1), outcome=["success"] * 3)["z_memory"]
    z_b = tme(feat, core, dA.unsqueeze(0).expand(3, -1, -1),
              dB.unsqueeze(0).expand(3, -1, -1), outcome=["failure"] * 3)["z_memory"]
assert not torch.allclose(z_a, z_b)
assert tme.num_trainable_parameters() > 0
print("TrainedMemoryEncoder: enriched shapes ok, params =",
      tme.num_trainable_parameters())
ok.append("memory_encoder: TrainedMemoryEncoder")

# OutcomeAwareAttention
att = OutcomeAwareAttention()
q = torch.randn(256)
ctx = torch.randn(4, 256)
err = torch.randn(4, 8)
dirm = torch.randn(4, 16)
vals = torch.randn(4, 32)
outs = ["success", "failure", "recovery", "failure"]
M, w = att.forward_single(q, ctx, err, dirm, vals, outs)
assert M.shape == (32,) and w.shape == (4,)
assert abs(float(w.sum()) - 1.0) < 1e-5
# batched forward
M2, w2 = att(torch.randn(2, 256), torch.randn(2, 4, 256),
             torch.randn(2, 4, 8), torch.randn(2, 4, 16),
             torch.randn(2, 4, 32),
             [["success", "failure", "recovery", "failure"]] * 2)
assert M2.shape == (2, 32) and w2.shape == (2, 4)
# outcome-aware: attention weights must differ when outcome distribution
# across the episode batch changes (softmax is shift-invariant, so use a
# mixed batch vs a uniform batch)
M_mix, w_mix = att.forward_single(q, ctx, err, dirm, vals,
                                  ["success", "failure", "recovery", "failure"])
M_unif, w_unif = att.forward_single(q, ctx, err, dirm, vals,
                                    ["success"] * 4)
assert not torch.allclose(w_mix, w_unif)
print("OutcomeAwareAttention: single + batched + outcome-sensitive ok")
ok.append("memory_encoder: OutcomeAwareAttention")

print("\nALL OK:", len(ok))
for o in ok:
    print("  -", o)
