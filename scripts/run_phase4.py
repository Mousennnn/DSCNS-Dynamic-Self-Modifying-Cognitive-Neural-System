"""Phase 4 experiment: learned model-driven structural self-adaptation.

Design-report modification proposal, final experiment (section 19):

    Rule-based controller vs Learned controller vs Fixed topology

On a shifted distribution stream (16 rounds: general(4) -> code(4) ->
mixed_code(4) -> science(4)) we compare three arms:

  * fixed    -- 5 networks, no structure evolution (control);
  * rule     -- rule-based controller ("single_rule"): one ArchitectureAction
                per round decided by the rule engine, executed through the
                candidate -> evaluate -> accept/rollback protocol;
  * learned  -- Stage A (8 rounds): rule actions drive + the policy imitates
                them; Stage B (8 rounds): the learned policy proposes actions,
                receives the modification reward (proposal section 11) and is
                updated with REINFORCE (proposal section 12).

Reported: performance curves, AF/FWT/CLS, structural metrics, modification
logs (op counts, acceptance rate, reward, policy entropy, imitation loss).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (DOMAINS, build_system, eval_per_domain_loss, make_base_model,
                    make_config, prepare_data, save_results)
from dscns.evaluation import (compute_continual_learning_metrics,
                              structural_metrics)
from dscns.utils import set_seed
import numpy as np


def make_shifted_stream(config, data, rng, phases=None):
    """general(4) -> code(4) -> mixed_code(4) -> science(4): shift stream."""
    train = data["train"]
    per_round = config.samples_per_round
    stream = []
    phases = phases or [("general", 4), ("code", 4), ("mixed_code", 4),
                        ("science", 4)]
    for phase, n in phases:
        for _ in range(n):
            if phase == "mixed_code":
                samples = []
                for domain, frac in [("code", 0.5), ("general", 0.25),
                                     ("science", 0.25)]:
                    pool = train[domain]
                    k = max(1, int(per_round * frac))
                    samples += [{"text": t, "domain": domain, "source": domain,
                                 "reliability": 0.8}
                                for t in rng.choice(pool, size=min(k, len(pool)),
                                                    replace=False)]
                rng.shuffle(samples)
                stream.append(samples[:per_round])
            else:
                pool = train[phase]
                k = min(per_round, len(pool))
                stream.append([{"text": t, "domain": phase, "source": phase,
                                "reliability": 0.8}
                               for t in rng.choice(pool, size=k, replace=False)])
    return stream


def run_arm(tag, controller, config, data, eval_sets, stream):
    t0 = time.time()
    config.evolution_controller = controller
    config.evolution_enabled = controller != "none"
    base = make_base_model(config, tag=f"p4_{tag}")
    system = build_system(config, base, data)
    matrix, spec_curve, nnet_curve = [], [], []
    for r, batch in enumerate(stream):
        info = system.process_experiences(batch)
        perf = eval_per_domain_loss(system, eval_sets, None, 48)
        matrix.append([perf[d] for d in DOMAINS])
        system.meta_update({"overall": float(np.mean(list(perf.values())))})
        sm = structural_metrics(system.networks, system.domain_embeddings)
        spec_curve.append(sm["mean_specialization"])
        nnet_curve.append(sm["n_networks"])
        tag_col = "learned" if controller == "learned" else tag
        extra = ""
        if getattr(system, "self_mod", None) is not None:
            tr = system.self_mod.trace[-1] if system.self_mod.trace else {}
            extra = (f" | op={tr.get('op')}({tr.get('source')})"
                     f" reward={tr.get('reward', '-')}")
        print(f"[{tag_col}] round {r + 1}: " +
              ", ".join(f"{d}={perf[d]:.4f}" for d in DOMAINS) +
              f" | nets={sm['n_networks']:.0f} spec={sm['mean_specialization']:.3f}"
              + extra, flush=True)

    results = {
        "tag": tag,
        "controller": controller,
        "performance_matrix": matrix,
        "final_performance": {d: float(matrix[-1][j])
                              for j, d in enumerate(DOMAINS)},
        "specialization_curve": spec_curve,
        "n_networks_curve": nnet_curve,
        "structure": structural_metrics(system.networks, system.domain_embeddings),
        "evolution_log": getattr(getattr(system, "evolver", None),
                                 "evolution_log", []),
        "wall_seconds": time.time() - t0,
    }
    results.update(compute_continual_learning_metrics(matrix, domains=DOMAINS))
    if getattr(system, "self_mod", None) is not None:
        sm_ctrl = system.self_mod
        results["modification"] = {
            "memory": sm_ctrl.memory.snapshot(),
            "trace": sm_ctrl.trace,
            "action_counts_rule": sm_ctrl.memory.action_counts(source="rule"),
            "action_counts_policy": sm_ctrl.memory.action_counts(source="policy"),
            "success_rate": sm_ctrl.memory.success_rate(),
            "mean_reward": sm_ctrl.memory.mean_reward(),
            "rewards": [r.reward for r in sm_ctrl.memory.rl_samples()
                        if r.reward is not None],
            "acceptance": [1.0 if r.accepted else 0.0
                           for r in sm_ctrl.memory.structural()],
            "imitation_loss_curve": [t.get("imitation_loss", 0.0)
                                     for t in sm_ctrl.trace],
            "rl_loss_curve": [t.get("rl_loss", 0.0) for t in sm_ctrl.trace],
            "entropy_curve": [t.get("policy_entropy", None)
                              for t in sm_ctrl.trace],
            "warmup_rounds": getattr(config, "learned_warmup_rounds", 8),
            "adaptation_window": getattr(config, "adaptation_window", 3),
        }
    # adaptation speed: code perf at round 4 (start of code phase) vs round 8
    pm = results["performance_matrix"]
    results["code_adapt_speed"] = {
        "round4": pm[3][DOMAINS.index("code")],
        "round8": pm[7][DOMAINS.index("code")],
        "delta": pm[7][DOMAINS.index("code")] - pm[3][DOMAINS.index("code")],
    }
    return results


def plot_comparison(all_results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for tag, res in all_results.items():
        pm = np.asarray(res["performance_matrix"])
        ax.plot(range(1, len(pm) + 1), pm.mean(axis=1), marker="o",
                markersize=3, label=tag)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean performance (exp(-loss))")
    ax.set_title("Overall performance per round")
    ax.legend()
    ax2 = axes[1]
    for tag, res in all_results.items():
        ax2.plot(range(1, len(res["n_networks_curve"]) + 1),
                 res["n_networks_curve"], marker="s", markersize=3, label=tag)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Number of networks")
    ax2.set_title("Network population size")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_actions(all_results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rule_res = all_results.get("rule", {})
    learned_res = all_results.get("learned", {})
    mods = ["no_op", "expand", "contract", "split", "merge",
            "connect", "disconnect"]
    rule_counts = rule_res.get("modification", {}).get("action_counts_rule", {})
    pol_counts = learned_res.get("modification", {}).get("action_counts_policy", {})
    r_vals = [rule_counts.get(m, 0) for m in mods]
    p_vals = [pol_counts.get(m, 0) for m in mods]
    x = np.arange(len(mods))
    w = 0.38
    ax.bar(x - w / 2, r_vals, w, label="rule controller (decided)")
    ax.bar(x + w / 2, p_vals, w, label="learned policy (Stage B decided)")
    ax.set_xticks(x)
    ax.set_xticklabels(mods, rotation=20)
    ax.set_ylabel("Count")
    ax.set_title("ArchitectureAction distribution: rule vs learned controller")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_reward(learned_res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mod = learned_res.get("modification", {})
    rewards = mod.get("rewards", [])
    acceptance = mod.get("acceptance", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if rewards:
        ax.plot(range(1, len(rewards) + 1), rewards, marker="o", markersize=4,
                label="reward")
        cum = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax.plot(range(1, len(rewards) + 1), cum, marker="s", markersize=3,
                label="cumulative mean")
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Modification #")
    ax.set_ylabel("Reward")
    ax.set_title("Modification reward (learned controller)")
    ax.legend()
    ax2 = axes[1]
    if acceptance:
        ax2.plot(range(1, len(acceptance) + 1), acceptance, marker="o",
                 markersize=4)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["rejected", "accepted"])
    ax2.set_xlabel("Structural modification #")
    ax2.set_title("Accept / reject per modification")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_learning(learned_res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mod = learned_res.get("modification", {})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ent = [e for e in mod.get("entropy_curve", []) if e is not None]
    if ent:
        ax.plot(range(1, len(ent) + 1), ent, marker="o", markersize=3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Policy action entropy (nats)")
    ax.set_title("Learned policy action entropy")
    ax2 = axes[1]
    il = mod.get("imitation_loss_curve", [])
    rl = mod.get("rl_loss_curve", [])
    if il:
        ax2.plot(range(1, len(il) + 1), il, marker="o", markersize=3,
                 label="imitation loss (Stage A)")
    if rl:
        ax2.plot(range(1, len(rl) + 1), rl, marker="s", markersize=3,
                 label="RL loss (Stage B)")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title("Policy learning curves")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--arms", default="fixed,rule,learned")
    args = ap.parse_args()

    config = make_config()
    config.samples_per_round = 32
    config.num_networks = 5
    config.seed = args.seed
    config.total_rounds = 16
    config.evolution_min_round = 3      # warm-up needs room for rule actions
    config.learned_warmup_rounds = args.warmup
    config.adaptation_window = args.window
    set_seed(config.seed)
    data = prepare_data(config)
    eval_sets = data["eval"]
    rng = np.random.RandomState(config.seed)
    stream = make_shifted_stream(config, data, rng)

    all_results = {}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for tag, ctrl in (("fixed", "none"), ("rule", "single_rule"),
                      ("learned", "learned")):
        if tag not in arms:
            continue
        all_results[tag] = run_arm(tag, ctrl, config, data, eval_sets, stream)
        save_results(args.out, tag, all_results[tag])

    save_results(args.out, "summary", all_results)

    os.makedirs(args.out, exist_ok=True)
    plot_comparison(all_results, os.path.join(args.out, "phase4_comparison.png"))
    if "rule" in all_results and "learned" in all_results:
        plot_actions(all_results, os.path.join(args.out, "phase4_actions.png"))
    if "learned" in all_results:
        plot_reward(all_results["learned"],
                    os.path.join(args.out, "phase4_reward.png"))
        plot_learning(all_results["learned"],
                      os.path.join(args.out, "phase4_learning.png"))

    print("=== Phase 4 summary ===")
    for tag, res in all_results.items():
        print(f"{tag}: final=" + ", ".join(f"{d}={v:.4f}" for d, v in
                                           res["final_performance"].items()))
        print(f"  AF={res['AF']:.4f} FWT={res['FWT']:.4f} CLS={res['CLS']:.4f}"
              f" | nets={res['structure']['n_networks']:.0f}"
              f" | code_adapt={res['code_adapt_speed']['delta']:+.4f}")
        if "modification" in res:
            m = res["modification"]
            print(f"  mods: {m['action_counts_rule']} (rule) "
                  f"{m['action_counts_policy']} (policy) | "
                  f"success_rate={m['success_rate']:.2f} "
                  f"mean_reward={m['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
