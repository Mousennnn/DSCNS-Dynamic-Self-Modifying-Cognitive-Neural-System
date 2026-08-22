"""v0.6.0 / Phase 6 Regression Tests."""
import os, sys, json, time, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS  {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {name} {detail}")


def test_policy_trace():
    print("\n--- Policy Trace ---")
    from dscns.policy_trace import PolicyTraceLog, PolicyTraceEntry
    log = PolicyTraceLog()
    for i in range(50):
        entry = PolicyTraceEntry(
            round_id=i+1, seed=42, condition="test",
            policy_target_probs=[0.6, 0.3, 0.1],
            policy_magnitude=0.5, policy_confidence=0.7,
            actual_target=i % 3, actual_magnitude=0.45,
            applied=True, outcome="success" if i % 4 == 0 else "failure",
            reward=0.1 if i % 4 == 0 else -0.1,
        )
        log.record(entry)
    diag = log.diagnostics()
    check("n_entries", diag["n_entries"] == 50)
    check("target_accuracy_range", 0 <= diag["target_accuracy"] <= 1)
    check("magnitude_correlation_range", -1 <= diag["magnitude_correlation"] <= 1)
    check("applied_ratio", 0.8 < diag["applied_ratio"] <= 1.0)
    check("outcome_distribution", len(diag["outcome_distribution"]) > 0)
    # trajectory
    traj = log.policy_trajectory()
    check("trajectory_length", len(traj) == 50)
    check("trajectory_has_round_id", "round_id" in traj[0])


def test_outcome_policy_learning():
    print("\n--- Outcome Policy Learning ---")
    from dscns.outcome_policy_learning import (
        OutcomeDirectedPolicyLearner, ModificationRewardModel, PolicyCreditAssigner)
    # reward model
    rm = ModificationRewardModel()
    rm.set_baseline(0.5, 100.0)
    for i in range(20):
        r = rm.compute_reward(
            round_id=i+1,
            performance_before=0.5 + np.random.randn()*0.05,
            performance_after=0.5 + np.random.randn()*0.05,
            param_norm_before=100.0 + i,
            param_norm_after=100.0 + i + np.random.randn(),
        )
        check(f"reward_total_is_float_{i}", isinstance(r.total, float))
    summary = rm.summary()
    check("reward_summary_n", summary["n_rewards"] == 20)
    # credit assigner
    ca = PolicyCreditAssigner(gamma=0.95, credit_horizon=5)
    ca.record_decision(1, target=0, magnitude=0.5, reward=0.1)
    ca.record_decision(2, target=1, magnitude=0.3, reward=-0.1)
    credits = ca.compute_all_credits({1: 0.1, 2: -0.1, 3: 0.05})
    check("credit_computed", len(credits) == 2)
    # full pipeline
    learner = OutcomeDirectedPolicyLearner()
    for i in range(10):
        learner.step(
            round_id=i+1,
            performance_before=0.5, performance_after=0.51,
            param_norm_before=100, param_norm_after=101,
            target=i % 3, magnitude=0.5)
    s = learner.summary()
    check("outcome_learner_rounds", s["n_rounds"] == 10)


def test_modification_guard():
    print("\n--- Modification Guard ---")
    from dscns.modification_guard import ModificationGuard
    guard = ModificationGuard(max_param_drift=100.0, max_param_norm=5000.0)
    # safe state
    for i in range(5):
        state = guard.check_safety(
            round_id=i+1, param_norm=100 + i, param_drift=i,
            policy_entropy=0.8, probe_performance=0.5)
        check(f"safe_magnitude_scale_{i}", state.magnitude_scale == 1.0)
    check("is_safe_initially", guard.is_safe())
    # apply guard
    mag = guard.apply_guard(0.5)
    check("apply_guard_returns_positive", mag > 0)
    # dangerous state
    state = guard.check_safety(
        round_id=100, param_norm=10000, param_drift=200,
        policy_entropy=0.01, probe_performance=0.001)
    check("dangerous_risk_high", state.risk_level > 0.5)
    check("dangerous_scale_reduced", state.magnitude_scale < 1.0)
    check("guard_never_zero", guard.apply_guard(0.5) > 0)
    # summary
    summary = guard.summary()
    check("guard_has_n_checks", summary["n_checks"] > 0)


def test_checkpoint_manager():
    print("\n--- Checkpoint Manager ---")
    from dscns.checkpoint_manager import CheckpointManager, CheckpointMetadata
    test_dir = tempfile.mkdtemp()
    try:
        mgr = CheckpointManager(test_dir, version="v0.6.0")
        state = {"weights": torch.randn(10), "bias": torch.randn(5)}
        # save best
        score = mgr.compute_best_score(performance=0.8, rfr=0.1, drift=1.0, stability=0.9)
        check("best_score_positive", score > 0)
        path = mgr.save_best(
            state_dict=state, condition="test", seed=42, round_id=100,
            score=score, score_components={"perf": 0.5})
        check("best_path_not_empty", len(path) > 0)
        check("best_file_exists", os.path.exists(os.path.join(path, "model.pt")))
        check("best_metadata_exists", os.path.exists(os.path.join(path, "metadata.json")))
        # verify
        check("verify_integrity", mgr.verify_integrity("best"))
        # load
        loaded = mgr.load_best()
        check("load_best_not_none", loaded is not None)
        check("load_best_has_weights", "weights" in loaded)
        # save final
        mgr.save_final(state_dict=state, condition="test", seed=42, round_id=450)
        final = mgr.load_final()
        check("save_final_load", final is not None)
        # artifact manifest
        manifest = mgr.artifact_manifest()
        check("manifest_has_artifacts", len(manifest["artifacts"]) >= 2)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_relay_manager():
    print("\n--- Relay Manager ---")
    from dscns.relay_manager import RelayManager, RelayLineage
    test_dir = tempfile.mkdtemp()
    try:
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
        check("relay_path_exists", os.path.exists(path))
        # load
        loaded = mgr.load_latest_relay()
        check("relay_load_not_none", loaded is not None)
        if loaded is not None:
            check("relay_has_model_state", "model_state" in loaded)
        # stages
        stages = mgr.available_stages()
        check("relay_has_stages", len(stages) > 0)
        # lineage
        line = mgr.get_lineage()
        check("lineage_source_version", line.source_version == "v0.5.3")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_config():
    print("\n--- Config ---")
    import yaml
    with open("config/phase6.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    check("config_version", config["version"] == "0.6.0")
    check("config_phase", config["phase"] == "P6")
    check("config_has_full_policy", "FullPolicy" in config["conditions"])
    check("config_has_oracle", "Oracle" in config["conditions"])
    check("config_has_random", "Random" in config["conditions"])
    check("config_11_conditions", len(config["conditions"]) == 11)


def test_divergence():
    print("\n--- Policy Divergence ---")
    # import from run_phase6
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from run_phase6 import compute_policy_divergence
    dist_a = [{0: 0.5, 1: 0.3, 2: 0.2}] * 10
    dist_b = [{0: 0.3, 1: 0.5, 2: 0.2}] * 10
    div = compute_policy_divergence(dist_a, dist_b)
    check("divergence_has_kl", "kl" in div)
    check("divergence_kl_positive", div["kl"] > 0)
    check("divergence_js_range", 0 <= div["js"] <= 1)
    check("divergence_cosine_range", -1 <= div["cosine"] <= 1)
    # identical
    div_same = compute_policy_divergence(dist_a, dist_a)
    check("divergence_same_kl_zero", div_same["kl"] < 0.01)


def test_adaptive_exploration():
    print("\n--- Adaptive Exploration ---")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from run_phase6 import adaptive_epsilon
    # high confidence -> low epsilon
    eps_high = adaptive_epsilon(0.9, 0.1, base_eps=0.15, min_eps=0.02)
    # low confidence -> high epsilon
    eps_low = adaptive_epsilon(0.2, 0.9, base_eps=0.15, min_eps=0.02)
    check("adaptive_high_conf_low_eps", eps_high < eps_low)
    check("adaptive_eps_in_range", 0.02 <= eps_high <= 0.30)
    check("adaptive_eps_in_range_low", 0.02 <= eps_low <= 0.30)


if __name__ == "__main__":
    print("DSCNS v0.6.0 / Phase 6 — Test Suite")
    print("=" * 60)

    tests = [
        test_policy_trace,
        test_outcome_policy_learning,
        test_modification_guard,
        test_checkpoint_manager,
        test_relay_manager,
        test_config,
        test_divergence,
        test_adaptive_exploration,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  ERROR in {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            FAIL_COUNT += 1

    print(f"\n{'='*60}")
    print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*60}")
    sys.exit(1 if FAIL_COUNT > 0 else 0)
