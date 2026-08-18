# DSCNS 设计报告完整复现报告（Phase 0 + Phase 1 + Phase 2 + Phase 3）

> 复现对象：《动态自修改认知网络系统设计报告》(DSCNS v1.0)
> 环境：conda `rtdetr`（Python 3.8.16 / PyTorch 1.13.1+cu117 / RTX 3070 Ti 8GB）
> 底座模型：GPT-2 small（124M 参数，位于设计报告 100M-500M 目标区间），本地冻结 + 每认知网络独立 LoRA(r=16) 适配器
> 数据：Wikitext-103（常识）、GSM8K（数学）、MATH-500（逻辑，原 hendrycks/competition_math 需授权，已自动回退）、HumanEval（代码）、SciQ（科学）

---

## 1. 复现范围与结果文件

| 内容 | 位置 |
|---|---|
| 设计报告原文存档 | `docs/DSCNS_design_report.md` |
| 系统实现（11 个模块） | `dscns/` |
| 实验脚本 | `scripts/run_phase1.py`、`run_phase2.py`、`run_phase3.py` |
| 实验配置 | `config/phase1.yaml` |
| 结果数据（JSON + 图） | `experiments/phase1|phase2|phase3/` |
| 指标对比表 | `experiments/comparison.md` |
| 本报告 | `REPORT_zh.md` |

## 2. 架构实现对照（设计报告章节 → 代码）

| 报告章节 | 实现 |
|---|---|
| §2.1 经验管理层 | `experience.py`：ExperienceBuffer（P1：接收≠学习）+ CandidateParser（批量解析为带嵌入/领域/不确定度的候选知识） |
| §2.1/§3.3 多网络认知层 | `networks.py`：5 个认知网络（N1 世界/ N2 数学/ N3 逻辑/ N4 语言/ N5 验证），共享冻结底座、独立 LoRA 适配器（Θ_i 独立、内存共享），四维评估 Q=(R,N,C,I) |
| §3.2 知识状态分级 | 每条知识记录 Level 0-3 与内化度 I_ij∈[0,1]（`knowledge_states` / `internalization_level`） |
| §3.4/§8.2 跨网络验证 | `verification.py`：信任加权聚合 C_final=Σw_iC_i/Σw_i（w_i=trust×R）、冲突检测（max-min>C_thr）、四策略冲突解决（历史证据/语义检索/延迟决策/证据重算）、信任权重动态更新（±0.05） |
| §3.5/§8.3 渐进内化 | `internalization.py`：试探更新（α 阶梯 0.2→1.0 × base_lr）→ 全领域探针回归测试 → 退化>tolerance 即回滚停止 → 渐进巩固 |
| §5 三层记忆 | `memory.py`：情景（带嵌入检索）、语义（轻量知识图谱，含 I_ij）、程序（任务→成功步骤），共享 MemorySystem |
| §6 元认知层 | `metacognition.py`：能力/不确定性/覆盖度/结构效率监控、自适应学习率、`Score=2·IG+U+R−0.5·C` 主动选择 |
| §8.1 通信协议 | `communication.py`：NetworkMessage + 10 类消息 + 异步总线（可追溯日志 + 共激活矩阵） |
| §4 结构演化 | `evolution.py`：专业化度、分裂（多样性+聚类+负迁移，k-means 划分 + 适配器克隆）、合并（功能重叠+共激活+表示相似）、连接 w=α·CoAct+β·InfoFlow |
| §7.3/§11 系统闭环 | `system.py`：DSCNSSystem.process_experiences 完整管道 |
| §7.4/§10.3 指标 | `evaluation.py`：AF / FWT / CLS / 各领域遗忘 / 获取-保留 / 结构指标 |

## 3. 实验设置

- 经验流（24 轮）：general(4) → math(4) → logic(4) → code(4) → science(4) → mixed(4)，每轮 32 条经验
- 计算公平性：三种模式每轮梯度步预算相同（8 步）。Control 无条件 8 步；Exp1 每轮最多 5 步渐进内化（回归门控）；Exp2 每轮最多 2 个网络各 5+3 步（门控）
- 指标：性能 = exp(−CE loss)（掩码逐 token 损失，避免 padding 污染）；另报告生成准确率
- 结构演化最小稳定期 6 轮；合并阈值 overlap>0.97、共激活≥8；每轮最多 1 次分裂/合并

## 4. Phase 1 结果（H1/H3/H4 验证）

| 指标 | Control | Exp1 | Exp2 |
|---|---|---|---|
| 平均遗忘 AF ↓ | 0.0037 | **0.0010** | 0.0013 |
| 前向迁移 FWT ↑ | 0.0002 | −0.0041 | **+0.0013** |
| 综合持续学习分 CLS ↑ | **0.0868** | 0.0735 | 0.0565 |
| 平均新知识获取 ↑ | 0.0942 | 0.0725 | 0.0548 |
| 平均旧知识保留 ↑ | **0.0905** | 0.0745 | 0.0578 |

各领域（final vs 峰值）遗忘：Control 在 science 阶段对 general/math/code 的遗忘最重（general −0.23、math −1.82、code −1.82），Exp1 明显更稳（−0.02/−0.45/−1.05），Exp2 最稳（+0.03/−0.12/−0.48，general 甚至无遗忘）。

**结论（诚实陈述）**：
- **H3（验证降低遗忘）部分成立**：选择性渐进内化显著降低灾难性遗忘（AF：Control 3.7e-3 → Exp1 1.0e-3 / Exp2 1.3e-3；各领域遗忘率均大幅收窄）。
- **H1（多网络综合优势）未完全成立**：Exp2 的前向迁移最优且唯一为正（+0.0013），但绝对性能（CLS）低于 Control——8 步/轮预算下 5 网络特化分工使每个网络训练轮次有限（≈5 轮/网络），绝对性能落后。原型尺度下"综合性能"未超越直接微调，需要更长训练周期验证。
- **H4 佐证**：Exp2 信任权重分化（N1=0.9, N4=0.8, N3=0.7, N2/N5=0.65）表明验证网络对"更可靠"的网络赋予了更高权重；760 次广播 + 3800 条置信度消息全程可追溯。

## 5. Phase 2 结果（主动学习）

| 策略 | 最终性能 | 知识覆盖度 |
|---|---|---|
| random（基线） | 0.0572 | 0.256 |
| uncertainty（不确定性） | 0.0466 | 0.256 |
| info_gain（信息增益） | **0.0591** | 0.256 |
| meta（元认知引导） | 0.0561 | 0.256 |

- 基于**学习器当前状态**（而非冻结底座）计算不确定度后，信息增益采样优于随机基线（+3.3%）；纯不确定性采样反而最差（选择过难样本）。
- 设计报告预期的 Exp3>Exp2>Exp1>Baseline 排序未完全复现：本原型中 meta 策略的"薄弱领域加权"在覆盖度普遍偏低时退化为近似信息增益，且增益幅度有限——需要更大经验池与更多轮次才能拉开差距。

## 6. Phase 3 结果（H2 验证：结构演化）

| 指标 | fixed（固定 5 网络） | evolve（动态演化） |
|---|---|---|
| 最终性能（5 域均值） | **0.0575** | 0.0533 |
| 代码域适应速度（round4→7 Δ） | **+0.0130** | +0.0041 |
| 最终网络数 | 5 | 5 |

演化事件：round7 合并（5→4，N2+N3 功能重叠）→ round9 分裂（→5）→ round11 再分裂（→6）→ round13 合并 + 动态连接建立（N4a–N2ab、N2ab–N1）。

**结论（诚实陈述）**：分裂/合并/连接机制全部按设计触发并记录（§4.1-4.4 全部落地），但 H2 未得到支持——结构操作在短期内引入了性能波动（正好实证了报告 §14.1 风险 2："分裂/合并操作可能引入短期性能波动"，缓解手段"强制稳定期"也已实现并生效）。固定结构在本分布漂移流上表现更好。要体现动态结构优势，需要更长的演化后稳定期与更大的分布漂移幅度。

## 7. 与设计报告的已知偏差

1. **MATH 数据集**：`hendrycks/competition_math` 为受限数据集（需授权），自动回退到公开的 `HuggingFaceH4/MATH-500`（500 题，逻辑域训练池 420 条）。
2. **HumanEval 规模小**（164 题）：代码域训练池仅 84 条，多轮复用。
3. **评估口径**：性能 = 掩码逐 token 的 exp(−loss)；初始版本曾将 padding 位置损失计入（导致数值系统性偏低），已修正并全量重跑。
4. **内化粒度**：报告 §8.3 为单知识项渐进内化；为计算可行，实现为"每轮每个网络接受的候选组"批量渐进内化（回归门控语义一致）。
5. **相关性信号**：纯冻结底座嵌入的余弦相似度区分度弱（文本嵌入在高维空间聚类紧密），已加入"领域标签匹配加分"增强（系统可见领域标签，属合法观察信息）。
6. **演化频率**：加入 6 轮稳定期、每轮最多 1 次操作、更保守阈值（报告 §14.1 风险 2 的缓解措施的显式化）。

## 8. 复现步骤

```bash
conda activate rtdetr   # Python 3.8 / torch 1.13.1+cu117（RT-DETR 环境，已配置）
cd dscns

# 依赖（已装；如需重装：scripts/resolve_deps.py 解析+断点续传下载 wheel，离线安装）
python scripts/resolve_deps.py && pip install --no-index --find-links wheelhouse -r requirements.txt

# 数据（已缓存 data/hf/；离线时可跳过）
# 如网络受限，请按你的代理设置配置 HTTP_PROXY / HTTPS_PROXY 环境变量
set HTTP_PROXY=http://127.0.0.1:7897 & set HTTPS_PROXY=http://127.0.0.1:7897
python -c "import sys; sys.path.insert(0,'.'); from scripts.common import make_config, prepare_data; print({k: len(v) for k, v in prepare_data(make_config())['train'].items()})"

# 实验
set HF_HUB_OFFLINE=1
python scripts/run_phase1.py --modes control exp1 exp2 --out experiments/phase1   # ~1h
python scripts/run_phase2.py --rounds 8 --pool-per-domain 50 --out experiments/phase2
python scripts/run_phase3.py --out experiments/phase3
python scripts/make_report.py   # 汇总 comparison.md + 图
```

## 9. 风险与局限（复现过程实证）

- 验证成本：每试探步都做全领域回归测试，使 Exp2 每轮耗时约为 Control 的 3-4 倍（报告风险 1 的实证）。
- 8GB 显存：batch 8 × seq 192 为安全上限；`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32` 缓解碎片化。
- 网络中途不可用：PyPI/HF 直连 TLS 不稳定，本机需配置本地代理（HTTP_PROXY/HTTPS_PROXY 指向本机代理端口）；pip 安装采用"自研解析器 + 断点续传 + 离线安装"方案（wheelhouse/，不入库）。
- 结构演化短期波动：已实证（Phase 3），与报告风险 2 一致。

## 10. Phase 4 附录：学习式结构自修改（Learned Structural Self-Adaptation）

按设计报告修改方案《DSCNS 自主神经结构自修改机制》实现：将"是否/如何/在哪/改多少"修改网络结构的**决策权**从规则引擎转移到可训练策略。

**实现要点**

- 新增 `dscns/self_modification.py`：`SelfStateEncoder`（系统自状态 47 维 → z_self）→ `SelfModificationPolicy`（P(动作)/P(目标)/magnitude/value 头）→ `ArchitectureAction`；新增 `dscns/modification_memory.py` 记录每次修改经验（状态、动作、接受/回滚、奖励及分量）。
- `StructureEvolver` 保留执行能力并降级为**硬安全约束层**（`validate_action`：预算、目标合法性、数据量等）；`system.evolve_structure()` 按 `evolution_controller` 分发：`rule`（Phase 3 原流程）/ `single_rule` / `learned` / `none`。
- 学习两阶段：**Stage A**（前 8 轮）规则决策 + 监督模仿；**Stage B**（后 8 轮）策略自行提议，经"候选结构 → 短适应窗口（3 轮）→ 回归评估 → 接受/回滚"，按
  `Reward = Δ性能 − λ1·遗忘 − λ2·参数增长 − λ3·计算 − λ4·不稳定` 反馈，用带价值基线的 REINFORCE 更新（ε=0.15 探索）。

**最终对比实验**（shifted stream 16 轮：general(4)→code(4)→mixed_code(4)→science(4)；详细设计与判据见 `docs/PHASE4.md`）

| 指标 | fixed | rule | learned |
|---|---|---|---|
| 最终平均性能（5 域） | 0.0554 | 0.0570 | **0.0585** |
| 平均遗忘 AF ↓ | **0.0000** | 0.0099 | 0.0029 |
| 前向迁移 FWT ↑ | 0.0005 | **0.0009** | 0.0007 |
| 持续学习分 CLS ↑ | 0.0553 | 0.0471 | **0.0556** |
| 代码域适应（r4→8 Δ） | +0.0224 | +0.0326 | **+0.0336** |
| 最终网络数 | 5 | 2 | 2 |
| 修改成功率 | — | 1.00 | 1.00 |
| 修改平均奖励 | — | — | −0.0031 |

- **rule 臂**：4 次结构操作（r3/r6/r9 合并、r12 分裂）全部接受；激进合并损害数学域保留（math 终值 0.0361），AF 升至 0.0099。
- **learned 臂**：Stage A 模仿损失 1.99→1.55；Stage B 中策略**自主**提出合并（r11，N1+N3，接受，奖励 −0.009），另有 4 次 no-op、2 次非法动作被安全层拦截；策略动作熵 1.94→1.77（行为随学习发生变化），平均奖励 −0.0031。

**结论（单种子、原型尺度）**：本运行中 learned 控制器取得最高最终平均性能、最高 CLS 与最佳代码域适应，且遗忘低于 rule 控制器；差异（≈0.003）**不构成统计显著结论**。策略在奖励偏负后比原始规则引擎更保守——这是合理的"从反馈中学习"的行为表现；实验证明了"自状态→策略→结构动作→候选→评估→奖励→策略更新"闭环可行，且规则引擎决策逻辑关闭后策略仍能自主产生结构修改（对应修改方案的 8 条判定标准，见 `docs/PHASE4.md` §6）。

---

**复现完成日期：** 2026-08-18（Phase 0–3）；2026-08-18（Phase 4 学习式结构自修改）
**结论：** 系统框架（经验→观察→评估→验证→内化→记忆→演化闭环）完整可运行；H3 与 Phase-2 信息增益采样得到支持，H1/H2 的原型尺度验证结果不支持（并给出原因）；Phase 4 实现了"自状态→学习式策略→结构动作→候选→评估→奖励→策略更新"的闭环（单种子原型演示）；所有实验数据与日志可复现、可追溯。
