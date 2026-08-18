"""Aggregate experiment results into a comparison report (markdown + plots).

Reads experiments/phase1|phase2|phase3/*.json and writes:
- experiments/comparison.md   (metric tables for the report)
- experiments/compare_*.png   (comparison plots)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

DOMAINS = ["general", "math", "logic", "code", "science"]


def load(phase: str, name: str):
    path = os.path.join("experiments", phase, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def phase1_section(out: list):
    out.append("## Phase 1: 持续学习对照实验（Control / Exp1 / Exp2）\n")
    out.append("| 指标 | Control (顺序微调) | Exp1 (单网络选择性内化) | "
               "Exp2 (多网络交叉验证) |")
    out.append("|---|---|---|---|")
    rows = {}
    for mode in ["control", "exp1", "exp2"]:
        r = load("phase1", mode)
        if r is None:
            continue
        m = r["metrics"]
        pd_ = r["per_domain"]
        ar = r["acquisition_retention"]
        rows[mode] = {
            "AF": m["AF"], "FWT": m["FWT"], "CLS": m["CLS"],
            "mean_acquisition": ar["mean_acquisition"],
            "mean_retention": ar["mean_retention"],
            "final_mean": float(np.mean(r["performance_matrix"][-1])),
        }
    keys = [("AF", "平均遗忘 (AF ↓)"), ("FWT", "前向迁移 (FWT ↑)"),
            ("CLS", "综合持续学习分 (CLS ↑)"),
            ("mean_acquisition", "平均新知识获取 (↑)"),
            ("mean_retention", "平均旧知识保留 (↑)"),
            ("final_mean", "最终平均性能 (↑)")]
    for k, label in keys:
        out.append(f"| {label} | " +
                   " | ".join(fmt(rows[m].get(k)) if m in rows else "-"
                              for m in ["control", "exp1", "exp2"]) + " |")
    out.append("")

    out.append("### 各领域遗忘率（final 性能 vs 峰值）\n")
    out.append("| 领域 | Control | Exp1 | Exp2 |")
    out.append("|---|---|---|---|")
    for d in DOMAINS:
        vals = []
        for mode in ["control", "exp1", "exp2"]:
            r = load("phase1", mode)
            vals.append(fmt(r["per_domain"][d]["forgetting_rate"]) if r else "-")
        out.append(f"| {d} | " + " | ".join(vals) + " |")
    out.append("")

    # decision / trust info for exp2
    r2 = load("phase1", "exp2")
    if r2:
        st = r2.get("structure", {})
        out.append("### Exp2 系统结构与验证统计\n")
        out.append(f"- 总线消息统计: `{st.get('bus_messages')}`")
        out.append(f"- 验证事件统计: `{st.get('verification')}`")
        out.append(f"- 最终信任权重: `{st.get('trust_weights')}`")
        out.append(f"- 结构指标: 网络数={st.get('n_networks')}, "
                   f"互补度={fmt(st.get('mean_complementarity'))}, "
                   f"冗余度={fmt(st.get('mean_redundancy'))}, "
                   f"专业化={fmt(st.get('mean_specialization'))}")
        out.append("")
    out.append("")


def phase2_section(out: list):
    out.append("## Phase 2: 主动学习实验（随机 / 不确定性 / 信息增益 / 元认知引导）\n")
    out.append("| 策略 | 最终性能 | 最终知识覆盖度 | 达到50%覆盖轮数 |")
    out.append("|---|---|---|---|")
    summary = load("phase2", "summary") or {}
    for s, r in summary.items():
        out.append(f"| {s} | {fmt(r.get('final_perf'))} | "
                   f"{fmt(r.get('final_coverage'))} | "
                   f"{r.get('rounds_to_50pct_coverage')} |")
    out.append("")


def phase3_section(out: list):
    out.append("## Phase 3: 结构演化实验（固定结构 vs 动态演化，分布漂移流）\n")
    out.append("| 指标 | fixed | evolve |")
    out.append("|---|---|---|")
    rows = {}
    for tag in ["fixed", "evolve"]:
        r = load("phase3", tag)
        if r is None:
            continue
        rows[tag] = {
            "final_mean": float(np.mean(list(r["final_performance"].values()))),
            "code_delta": r.get("code_adapt_speed", {}).get("delta"),
            "spec": r.get("structure", {}).get("mean_specialization"),
            "n_nets": r.get("structure", {}).get("n_networks"),
            "log": r.get("evolution_log", []),
        }
    def get(tag, key, nd=4, default="-"):
        r = rows.get(tag)
        if r is None:
            return default
        v = r[key]
        return default if v is None else fmt(v, nd)
    out.append(f"| 最终性能 (5域均值) | {get('fixed', 'final_mean')} | "
               f"{get('evolve', 'final_mean')} |")
    out.append(f"| 代码域适应速度 (round4→7 Δ) | "
               f"{get('fixed', 'code_delta')} | {get('evolve', 'code_delta')} |")
    out.append(f"| 平均专业化度 | {get('fixed', 'spec')} | {get('evolve', 'spec')} |")
    out.append(f"| 最终网络数 | {get('fixed', 'n_nets', 0)} | "
               f"{get('evolve', 'n_nets', 0)} |")
    out.append("")
    log = rows.get("evolve", {}).get("log", [])
    if log:
        out.append("演化事件记录:\n")
        for e in log:
            out.append(f"- `{e}`")
        out.append("")


def phase4_section(out: list):
    out.append("## Phase 4: 学习式结构自修改实验（fixed vs rule vs learned）\n")
    out.append("| 指标 | fixed | rule | learned |")
    out.append("|---|---|---|---|")
    rows = {}
    for tag in ["fixed", "rule", "learned"]:
        r = load("phase4", tag)
        if r is None:
            continue
        m = r.get("modification", {})
        rows[tag] = {
            "final_mean": float(np.mean(list(r["final_performance"].values()))),
            "AF": r.get("AF"), "FWT": r.get("FWT"), "CLS": r.get("CLS"),
            "code_delta": r.get("code_adapt_speed", {}).get("delta"),
            "spec": r.get("structure", {}).get("mean_specialization"),
            "n_nets": r.get("structure", {}).get("n_networks"),
            "success": m.get("success_rate"),
            "mean_reward": m.get("mean_reward"),
        }
    def get(tag, key, nd=4, default="-"):
        rr = rows.get(tag)
        if rr is None:
            return default
        v = rr[key]
        return default if v is None else fmt(v, nd)
    out.append(f"| 最终性能 (5域均值) | {get('fixed','final_mean')} | "
               f"{get('rule','final_mean')} | {get('learned','final_mean')} |")
    out.append(f"| 平均遗忘 AF ↓ | {get('fixed','AF')} | {get('rule','AF')} | "
               f"{get('learned','AF')} |")
    out.append(f"| 前向迁移 FWT ↑ | {get('fixed','FWT')} | {get('rule','FWT')} | "
               f"{get('learned','FWT')} |")
    out.append(f"| 综合持续学习分 CLS ↑ | {get('fixed','CLS')} | "
               f"{get('rule','CLS')} | {get('learned','CLS')} |")
    out.append(f"| 代码域适应 (round4→8 Δ) | {get('fixed','code_delta')} | "
               f"{get('rule','code_delta')} | {get('learned','code_delta')} |")
    out.append(f"| 最终网络数 | {get('fixed','n_nets',0)} | "
               f"{get('rule','n_nets',0)} | {get('learned','n_nets',0)} |")
    out.append(f"| 结构修改成功率 | - | {get('rule','success',2)} | "
               f"{get('learned','success',2)} |")
    out.append(f"| 结构修改平均奖励 | - | {get('rule','mean_reward')} | "
               f"{get('learned','mean_reward')} |")
    out.append("")
    lr = rows.get("learned", {})
    if lr and "success" in lr:
        learned = load("phase4", "learned")
        if learned and learned.get("modification"):
            m = learned["modification"]
            out.append("学习式控制器 (learned) 修改记录统计:\n")
            out.append(f"- 动作分布 (rule 阶段): `{m.get('action_counts_rule')}`")
            out.append(f"- 动作分布 (policy 阶段): `{m.get('action_counts_policy')}`")
            out.append(f"- 修改奖励序列: `{[round(float(x),4) for x in m.get('rewards',[])]}`")
            out.append("")


def phase5_section(out: list):
    out.append("## Phase 5: 内生式参数自修改实验（fixed vs p5b + 负对照）\n")
    out.append("| 指标 | fixed | p5b (intrinsic) | random | constant | shuffled |")
    out.append("|---|---|---|---|---|---|")
    rows = {}
    for tag in ["fixed", "p5b", "p5c", "random", "constant", "shuffled"]:
        r = load("phase5", tag)
        if r is None:
            continue
        cl = r.get("closed_loop", {})
        rows[tag] = {
            "final_mean": float(np.mean(list(r["final_performance"].values()))),
            "AF": r.get("AF"), "FWT": r.get("FWT"), "CLS": r.get("CLS"),
            "triggers": r.get("triggers", 0),
            "accept": r.get("acceptance_rate"),
            "d_norm": cl.get("delta_norm_mean"),
            "pred": cl.get("pred_change_mean"),
        }
    def get(tag, key, nd=4, default="-"):
        rr = rows.get(tag)
        if rr is None:
            return default
        v = rr[key]
        return default if v is None else fmt(v, nd)
    cols = ["fixed", "p5b", "random", "constant", "shuffled"]
    out.append(f"| 最终性能 (5域均值) | " +
               " | ".join(get(t, "final_mean") for t in cols) + " |")
    out.append(f"| 平均遗忘 AF ↓ | " + " | ".join(get(t, "AF") for t in cols) + " |")
    out.append(f"| 前向迁移 FWT ↑ | " + " | ".join(get(t, "FWT") for t in cols) + " |")
    out.append(f"| 综合持续学习分 CLS ↑ | " + " | ".join(get(t, "CLS") for t in cols) + " |")
    out.append(f"| 可塑性触发次数 | " +
               " | ".join(get(t, "triggers", 0) for t in cols) + " |")
    out.append(f"| 修改接受率 | " +
               " | ".join(get(t, "accept", 2) for t in cols) + " |")
    out.append(f"| Δθ 平均范数 | " +
               " | ".join(get(t, "d_norm") for t in cols) + " |")
    out.append(f"| 预测变化率 | " +
               " | ".join(get(t, "pred") for t in cols) + " |")
    out.append("")
    out.append("> P5 核心命题是闭环存在性 `θ → h → Δθ → θ'`，性能为描述性指标；"
               "闭环证据（Test 1-6 全过 + 负对照区分）见 docs/PHASE5.md §5。"
               "P5-C（离线可塑性学习）因含额外适应计算（预算不对等）单独报告于 "
               "docs/PHASE5.md §5.4。")
    out.append("")


def main():
    out = ["# DSCNS 复现实验结果汇总\n",
           "> 基础模型: GPT-2 (124M) 本地冻结底座 + 每网络 LoRA(r=16) 适配器 | "
           "RTX 3070 Ti 8GB | 24 轮经验流: general→math→logic→code→science→mixed\n"]
    phase1_section(out)
    phase2_section(out)
    phase3_section(out)
    phase4_section(out)
    phase5_section(out)

    text = "\n".join(out)
    with open("experiments/comparison.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Phase1: per-mode per-domain curves + final comparison bar
    modes = ["control", "exp1", "exp2"]
    res = {m: load("phase1", m) for m in modes}
    if all(res.values()):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        for ax, m in zip(axes, modes):
            pm = np.asarray(res[m]["performance_matrix"])
            for j, d in enumerate(DOMAINS):
                ax.plot(range(1, len(pm) + 1), pm[:, j], label=d, marker="o",
                        markersize=2)
            ax.set_title(m)
            ax.set_xlabel("round")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("performance (exp(-loss))")
        fig.suptitle("Phase 1: per-domain performance over 24 rounds")
        fig.tight_layout()
        fig.savefig("experiments/phase1_comparison_curves.png", dpi=140)
        plt.close(fig)

        # AF / CLS bars
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [res[m]["metrics"]["AF"] for m in modes]
        cls = [res[m]["metrics"]["CLS"] for m in modes]
        x = np.arange(len(modes))
        w = 0.35
        ax.bar(x - w / 2, names, w, label="AF (lower better)")
        ax.bar(x + w / 2, cls, w, label="CLS (higher better)")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.set_title("Phase 1: continual-learning metrics")
        fig.tight_layout()
        fig.savefig("experiments/phase1_metrics.png", dpi=140)
        plt.close(fig)

    # Phase2: perf & coverage curves
    p2 = load("phase2", "summary") or {}
    if p2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for s, r in p2.items():
            axes[0].plot(range(1, len(r["perf_curve"]) + 1), r["perf_curve"],
                         marker="o", markersize=3, label=s)
            axes[1].plot(range(1, len(r["coverage_curve"]) + 1),
                         r["coverage_curve"], marker="o", markersize=3, label=s)
        axes[0].set_title("Phase 2: learning efficiency (perf/round)")
        axes[1].set_title("Phase 2: knowledge coverage/round")
        for ax in axes:
            ax.set_xlabel("round")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("experiments/phase2_curves.png", dpi=140)
        plt.close(fig)

    # Phase3: code-domain adaptation curves
    p3 = {t: load("phase3", t) for t in ["fixed", "evolve"]}
    if all(p3.values()):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for tag, r in p3.items():
            pm = np.asarray(r["performance_matrix"])
            ax.plot(range(1, len(pm) + 1), pm[:, DOMAINS.index("code")],
                    marker="o", markersize=3, label=tag)
        ax.set_xlabel("round")
        ax.set_ylabel("code-domain performance")
        ax.set_title("Phase 3: code domain under distribution shift (rounds 4-7)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("experiments/phase3_code_curve.png", dpi=140)
        plt.close(fig)

    print("\n[report] experiments/comparison.md + plots written")


if __name__ == "__main__":
    main()
