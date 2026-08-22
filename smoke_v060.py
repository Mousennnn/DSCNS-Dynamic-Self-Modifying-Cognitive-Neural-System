"""Smoke tests for v0.6.0 modules."""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_policy_trace():
    from dscns.policy_trace import PolicyTraceLog, PolicyTraceEntry
    log = PolicyTraceLog()
    for i in range(20):
        entry = PolicyTraceEntry(
            round_id=i+1, seed=42, condition="test",
            policy_target_probs=[0.5, 0.3, 0.2],
            policy_magnitude=0.5, policy_confidence=0.7,
            actual_target=i % 3, actual_magnitude=0.4 + np.random.randn()*0.1,
            applied=True, outcome="success" if i % 3 == 0 else "failure",
            reward=0.1 if i % 3 == 0 else -0.1,
        )
        log.record(entry)
    diag = log.diagnostics()
    assert diag["n_entries"] == 20
    assert 0 <= diag["target_accuracy"] <= 1
    assert -1 <= diag["magnitude_correlation"] <= 1
    print(f"  PolicyTrace: OK (target_acc={diag['target_accuracy']:.3f}, "
          f"mag_corr={diag['magnitude_correlation']:.3f})")


def test_outcome_policy_learning():
    from dscns.outcome_policy_learning import (
        OutcomeDirectedPolicyLearner, ModificationRewardModel, PolicyCreditAssigner)
    learner = OutcomeDirectedPolicyLearner()
    for i in range(10):
        reward = learner.step(
            round_id=i+1,
            performance_before=0.5 + np.random.randn()*0.05,
            performance_after=0.5 + np.random.randn()*0.05,
            param_norm_before=100.0 + i,
            param_norm_after=100.0 + i + np.random.randn(),
            target=i % 3, magnitude=0.5)
        assert isinstance(reward.total, float)
    summary = learner.summary()
    assert summary["n_rounds"] == 10
    print(f"  OutcomePolicyLearning: OK (n_rounds={summary['n_rounds']}, "
          f"mean_reward={summary['reward_model']['mean_total']:.4f})")


def test_modification_guard():
    from dscns.modification_guard import ModificationGuard
    guard = ModificationGuard(max_param_drift=100.0)
    for i in range(10):
        state = guard.check_safety(
            round_id=i+1, param_norm=100.0 + i*5,
            param_drift=i*3, policy_entropy=0.8,
            probe_performance=0.5)
        assert state.magnitude_scale > 0
    assert guard.is_safe()
    # test dangerous case
    state = guard.check_safety(
        round_id=100, param_norm=10000, param_drift=200,
        policy_entropy=0.05, probe_performance=0.001)
    assert state.magnitude_scale < 1.0
    print(f"  ModificationGuard: OK (mean_risk={guard.summary()['mean_risk']:.3f}, "
          f"n_interventions={guard.summary()['n_interventions']})")


def test_checkpoint_manager():
    from dscns.checkpoint_manager import CheckpointManager, CheckpointMetadata
    test_dir = os.path.join("experiments", "_test_ckpt")
    os.makedirs(test_dir, exist_ok=True)
    mgr = CheckpointManager(test_dir, version="v0.6.0")
    state = {"weights": torch.randn(10)}
    meta = CheckpointMetadata(condition="test", seed=42, round=100, score=0.5)
    path = mgr.save_checkpoint(state, meta, "best")
    assert os.path.exists(os.path.join(path, "model.pt"))
    assert os.path.exists(os.path.join(path, "metadata.json"))
    assert mgr.verify_integrity("best")
    loaded = mgr.load_best()
    assert loaded is not None
    # cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"  CheckpointManager: OK (save/load/verify passed)")


def test_relay_manager():
    from dscns.relay_manager import RelayManager, RelayLineage
    test_dir = os.path.join("experiments", "_test_relay")
    mgr = RelayManager(base_dir=test_dir, version="v0.6.0")
    mgr.lineage = RelayLineage(
        source_version="v0.5.3", target_version="v0.6.0",
        continued_rounds=0, total_lineage_rounds=450)
    relay_state = {
        "model_state": {"layer.weight": torch.randn(5, 5)},
        "round_counter": {"round": 450},
        "config": {"test": True},
    }
    path = mgr.save_relay(relay_state, condition="FullPolicy", seed=42, round_id=450)
    assert os.path.exists(path)
    loaded = mgr.load_latest_relay()
    assert loaded is not None
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"  RelayManager: OK (save/load passed)")


def test_config():
    import yaml
    with open("config/phase6.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["version"] == "0.6.0"
    assert config["phase"] == "P6"
    assert "FullPolicy" in config["conditions"]
    assert "Oracle" in config["conditions"]
    assert "Random" in config["conditions"]
    print(f"  Config: OK ({len(config['conditions'])} conditions defined)")


if __name__ == "__main__":
    print("v0.6.0 Smoke Tests")
    print("=" * 50)
    tests = [
        test_policy_trace,
        test_outcome_policy_learning,
        test_modification_guard,
        test_checkpoint_manager,
        test_relay_manager,
        test_config,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
