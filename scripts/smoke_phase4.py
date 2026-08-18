"""Phase 4 smoke test: learned self-modification machinery end-to-end."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dscns.config import DSCNSConfig
from dscns.evolution import StructureEvolver
from dscns.self_modification import (ArchitectureAction, FEAT_DIM, STATE_DIM,
                                     SelfModificationController)
from dscns.system import DSCNSSystem


def _cfg():
    return DSCNSConfig(
        model_name=os.path.join("models", "hf", "gpt2"),
        cache_dir="models/hf",
        num_networks=5,
        samples_per_round=8,
        max_grad_steps_per_round=4,
        max_len=96,
        evolution_enabled=True,
        evolution_controller="learned",
        evolution_min_round=0,
        learned_warmup_rounds=1,
        adaptation_window=1,
        modification_budget_max=8,
        total_rounds=6,
    )


def main():
    cfg = _cfg()
    from common import make_base_model, prepare_data

    data = prepare_data(cfg)
    base = make_base_model(cfg, tag="smoke_p4")
    exemplars = {d: t[:8] for d, t in data["train"].items()}
    probe = {d: t[:4] for d, t in data["probe"].items()}
    system = DSCNSSystem(base, cfg, exemplars, probe, seed=42)
    system.set_eval_sets({d: t[:8] for d, t in data["eval"].items()})
    sm = system.self_mod
    print("[1] controller created:", type(sm).__name__)

    # ---- state collection ----
    perf = system.best_domain_performance(system.eval_sets, 8)
    sm.track_perf(perf)
    probe_val = system.probe_perf()
    sm.track_probe(probe_val)
    s, F_net, net_ids, F_dom, dom_ids = sm.collect_state(system)
    print("[2] state dims:", s.shape, F_net.shape, F_dom.shape,
          "| net_ids:", net_ids, "| dom_ids:", dom_ids)
    assert s.shape == (STATE_DIM,), s.shape
    assert F_net.shape == (5, FEAT_DIM), F_net.shape

    # ---- policy act ----
    act = sm.propose(system, (s, F_net, net_ids, F_dom, dom_ids))
    print("[3] policy action:", act.operation, act.target, act.secondary_target,
          f"mag={act.magnitude:.3f} conf={act.confidence:.3f}")
    assert act.operation in [a for a in
        ["no_op", "expand", "contract", "split", "merge", "connect", "disconnect"]]

    # ---- validation + executor (connect) ----
    evolver = StructureEvolver()
    ok, reason = evolver.validate_action(
        act, system.networks, system.connections, budget=8, min_accepted=8,
        domains=list(system.domain_embeddings.keys()))
    print("[4] validate:", ok, reason)
    conn_act = ArchitectureAction("connect", target=net_ids[0],
                                  secondary_target=net_ids[1], magnitude=0.6,
                                  source="policy", round=0)
    ok2, r2 = evolver.validate_action(conn_act, system.networks,
                                      system.connections, domains=dom_ids)
    assert ok2, r2
    new_nets, new_conns, created = evolver.execute_action(
        conn_act, system.networks, system.connections, base.peft_model, 0)
    print("[5] connect executed:", new_conns)
    assert (net_ids[0], net_ids[1]) in new_conns

    # ---- snapshot / restore roundtrip (expand) ----
    snap = system._snapshot_architecture()
    exp_act = ArchitectureAction("expand", target="math", magnitude=0.5,
                                 source="policy", round=0)
    new_nets2, new_conns2, created2 = evolver.execute_action(
        exp_act, system.networks, system.connections, base.peft_model, 0,
        serial=1, domain_embeddings=system.domain_embeddings,
        memory=system.memory, base_lr=5e-4, network_factory=system._network_factory)
    print("[6] expand executed:", created2, "| nets:", len(new_nets2))
    assert "NX1" in new_nets2
    system.networks = new_nets2
    system.connections = new_conns2
    system.bus.networks = new_nets2
    system._restore_architecture(snap)
    print("[7] restore:", sorted(system.networks.keys()), "| conns:", system.connections)
    assert "NX1" not in system.networks
    assert sorted(system.networks.keys()) == sorted(net_ids)

    # ---- reward computation ----
    reward, comps = sm.compute_reward(
        "split", probe_val, probe_val + 0.01, 0.05, 0.06,
        1000, 1100, [probe_val, probe_val + 0.01], 1)
    print("[8] reward:", round(reward, 4), comps)

    # ---- imitation + RL training with synthetic data ----
    for _ in range(3):
        sm.record_imitation((s, F_net, net_ids, F_dom, dom_ids),
                            ArchitectureAction("no_op", source="rule", round=0))
    il = sm.train_imitation()
    print("[9] imitation loss:", round(il, 4))
    from dscns.modification_memory import ModificationRecord
    sm.memory.add(ModificationRecord(round=0, op="no_op", source="policy",
                                     state=s, accepted=True, reward=0.0))
    sm.memory.add(ModificationRecord(round=0, op="split", source="policy",
                                     state=s, accepted=True, reward=0.05))
    rl = sm.train_rl()
    print("[10] rl loss:", round(rl, 4))

    # ---- end-to-end: 2 full rounds with the learned controller ----
    rng = np.random.RandomState(0)
    stream = []
    for dom in ["general", "code"]:
        pool = data["train"][dom]
        k = min(8, len(pool))
        stream.append([{"text": t, "domain": dom, "source": dom, "reliability": 0.8}
                       for t in rng.choice(pool, size=k, replace=False)])
    for r, batch in enumerate(stream):
        info = system.process_experiences(batch)
        print(f"[11] round {r + 1}: evolutions={info['evolution']}")
    assert len(sm.trace) >= 2
    print("[12] trace rounds:", [t["round"] for t in sm.trace],
          "| last op:", sm.trace[-1]["op"], sm.trace[-1]["source"])
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
