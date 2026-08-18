# DSCNS — Dynamic Self-Modifying Cognitive Network System（動的自己修正認知ネットワークシステム）

> **ステータス：初期研究プロトタイプ（v0.3.0）**

DSCNS は、**継続学習**・**候補知識の検証**・**選択的内化**・**記憶**・
**メタ認知**・**構造進化**を探求する実験的研究プロトタイプです。

システムは次の概念ループを中心に設計されています：

```
経験
  → マルチネットワーク観察
  → 独立評価
  → ネットワーク間検証
  → メタ認知的意思決定
  → 選択的・漸進的内化
  → 回帰テスト
  → 記憶更新
  → 構造適応
  → 継続学習
```

本リポジトリには、現在のプロトタイプ実装とその実験結果（Phase 0–3）が含まれ
ます。これはオープンで誠実かつ再現可能な研究アーティファクトとして位置づけ
られており、継続学習・AGI・人間レベルの知能を解決したという主張では**ありま
せん**。

- [English](README.md) · [简体中文](README.zh-CN.md)

---

## 概要

DSCNS は単一の「データ → 勾配更新」フローを、閉ループの認知プロセスに置き換
えます：経験は複数ネットワークによって観察され、独立に評価され、ネットワーク
間で検証され、その後にのみ漸進的に内化されるか、呼び出し可能な知識として保存
されます。ネットワークトポロジ自体も学習可能な構造（分割・統合・接続）として
扱われます。Phase 4 では、いつ・何を・どこを・どれだけ修正するかの*決定*を
小さなニューラル自己修正ポリシーが模倣＋強化学習で学習し、Phase 5 では各
ネットワークの内部状態が自身のパラメータへの連続的な変更を直接生成します
（`θ → h → Δθ → θ'`、[docs/PHASE5.md](docs/PHASE5.md) 参照）。

現在のプロトタイプは、**凍結された GPT-2 small（124M）ベースモデル**と
**認知ネットワークごとの LoRA アダプタ**を基盤とし、設計文書に記述された
検証・内化・記憶・メタ認知・進化のスタック全体を実装しています。

## 中核アイデア

| 従来のパラダイム | DSCNS のパラダイム |
|---|---|
| データ → モデル → 勾配降下 → 固定パラメータ | 経験 → 複数ネットワーク観察 → 独立評価 → ネットワーク間検証 → 選択的内化 → 構造再編成 → 継続進化 |

10 の設計原則（要約）：情報を受け取ること ≠ 学習；学習 ≠ 即時のパラメータ
変更；パラメータ更新は検証を経なければならない；同じ経験を複数ネットワーク
が観察できる；観察 ≠ 内化；知識は共有も局所内化も可能；忘却は局所的・漸進的・
再活性化可能であるべき；ネットワークは相互通信・訂正・新接続形成ができる；
ネットワーク構造自体が学習結果である；システムは経験を通じて絶えず自己を
変化させる。

## システムアーキテクチャ

```
        外部環境 / 経験ストリーム
              │
        経験管理層
  （経験バッファ → 候補解析 → 能動的選択）
              │ 解析済み候補
              ▼
      マルチネットワーク認知層（N1 … N5）
  （共有凍結ベース + 独立 LoRA アダプタ）
              │ 評価 Q_i = (R, N, C, I)
              ▼
       ネットワーク間検証層
  （信頼重み付き集約・衝突検出/解決）
              │ 検証済み知識
              ▼
         メタ認知制御層
  （更新決定・適応的学習率・構造進化）
              │
              ▼
         記憶・知識ストア
  （エピソード・セマンティック・手続き）
```

## 主要メカニズム

### マルチネットワーク表現

複数の認知ネットワークが 1 つの凍結ベースモデルを共有しつつ、**独立した
LoRA アダプタ**を保持します——すなわち分散パラメータ空間

```
Θ_t = {θ_1, θ_2, …, θ_k}
```

ネットワークごとにパラメータは独立、ベース重みは共有（メモリ効率的）。

### 候補知識

新しい経験はパラメータを直接変更**しません**。まず*候補知識*となり、評価・
検証・内化決定を経ます。

### 独立評価

各ネットワークは候補を 4 次元で評価します：

```
Q_i = (R, N, C, I)
```

- **R** — 関連性（ネットワークのドメインとの類似度）
- **N** — 新規性（既内化知識との非類似度）
- **C** — 信頼度（ベースモデルの証拠 × 情報源の信頼性）
- **I** — 重要度（関連性 × 不確実性に基づく効用）

### ネットワーク間検証

ネットワークごとの信頼度を信頼重みで集約します：

```
w_i = trust_i × R_i，        C_final = Σ w_i·C_i / Σ w_i
```

衝突（`max(C_i) − min(C_i) > 閾値`）は証拠に基づく解決をトリガーし、証拠が
不十分な場合は強制的に回答せず**延期（defer）**します。ネットワークの信頼
重みは観測された正しさから動的に更新されます。

### 漸進的内化

知識はゲート付きループでネットワークに入ります：

```
試行更新 → 回帰テスト → 受諾 / ロールバック
```

さらに更新予算制約

```
‖Δθ‖₂ ≤ ε·‖θ‖₂
```

により、単一の知識項目がネットワークを急激に変化させないようにします。

### 知識レベル

プロトタイプは知識項目ごとの内化レベルを記録します：

| レベル | 状態 |
|---|---|
| Level 1 | **ブロードキャスト観察** — 知識の存在をシステムが把握 |
| Level 2 | **呼び出し可能な知識** — セマンティック記憶に保存、照会可能 |
| Level 3 | **内化された知識** — ネットワークのパラメータに影響 |

### 記憶システム

3 つの機能的な記憶層：

- **エピソード記憶** — 検索可能な時系列の生経験
- **セマンティック記憶** — 軽量な知識グラフ的表現
- **手続き記憶** — タスク種別ごとの成功行動系列

アーキテクチャはエピソード・セマンティック・手続き記憶の*機能的な区別に
着想を得ています*。ヒトの脳機構を模倣すると主張するものでは**ありません**。

## 現在の実装

| コンポーネント | モジュール | 設計レポート節 |
|---|---|---|
| 経験バッファ / 候補解析 | `dscns/experience.py` | §2.1, §3.1 |
| 認知ネットワーク + Q=(R,N,C,I) | `dscns/networks.py` | §2.1, §3.2–3.3 |
| ネットワーク間検証 | `dscns/verification.py` | §3.4, §8.2 |
| 漸進的内化 | `dscns/internalization.py` | §3.5, §8.3 |
| メタ認知コントローラ | `dscns/metacognition.py` | §6 |
| メッセージバス / 通信 | `dscns/communication.py` | §8.1 |
| 3 層記憶 | `dscns/memory.py` | §5 |
| 構造進化 | `dscns/evolution.py` | §4 |
| 学習型自己修正（Phase 4） | `dscns/self_modification.py` | Phase-4 提案 §3–12 |
| 修正記憶（Phase 4） | `dscns/modification_memory.py` | Phase-4 提案 §13 |
| 内在的可塑性（Phase 5） | `dscns/intrinsic_plasticity.py` | Phase-5 報告 §4–6 |
| 可塑性トレーナ（Phase 5-C） | `dscns/plasticity_trainer.py` | Phase-5 報告 §11 |
| システム統合 | `dscns/system.py` | §7.3, §11 |
| 指標（AF/FWT/CLS） | `dscns/evaluation.py` | §7.4, §10.3 |

## 実験設定

- **ベースモデル：** GPT-2 small（124M）、凍結、ローカルコピー
- **ネットワーク：** 5 つの認知ネットワーク（N1 世界 / N2 数学 / N3 論理 /
  N4 言語 / N5 検証）、各 LoRA（r=16）アダプタ
- **経験ストリーム：** 24 ラウンド = general(4) → math(4) → logic(4) →
  code(4) → science(4) → mixed(4)、1 ラウンド 32 経験
- **予算の対等性：** Control / Exp1 / Exp2 とも 1 ラウンド 8 勾配ステップ
  （Control：無条件；Exp1/Exp2：回帰ゲート付き）
- **性能指標：** exp(−マスク済み CE 損失)；生成精度は別途報告
- **ハードウェア：** NVIDIA RTX 3070 Ti 8GB（元実験）

## データセット

| ドメイン | データセット | 出典 |
|---|---|---|
| general | Wikitext-103（Wikipedia） | HuggingFace `wikitext` |
| math | GSM8K | HuggingFace `gsm8k` |
| logic | MATH-500（フォールバック） | HuggingFace `HuggingFaceH4/MATH-500` |
| code | HumanEval | HuggingFace `openai/openai_humaneval` |
| science | SciQ | HuggingFace `allenai/sciq` |

> 注：`hendrycks/competition_math` はデータセットの利用承認が必要なため、
> ローダーは自動的に公開の MATH-500 へフォールバックします。

**モデル重みとデータセットキャッシュは意図的にバージョン管理から除外されて
います。** 提供スクリプトで自動ダウンロードできます（[再現](#再現)参照）。

## 実験結果

詳細な数値は `experiments/comparison.md` と `REPORT_zh.md`（中国語の再現
レポート）にあります。要約：

| Phase | 実験 | 結果 |
|---|---|---|
| Phase 1 | マルチネットワーク継続学習 | 混合——Control より一貫して良いわけではない |
| Phase 1 | 漸進的内化（Exp1/Exp2） | **忘却の低減**（AF 0.0037 → 0.0010 / 0.0013） |
| Phase 1 | 前方転移（Exp2） | 検証した方式で唯一の正値（+0.0013） |
| Phase 2 | 情報利得サンプリング | 検証した戦略中最良（0.0591 vs ランダム 0.0572） |
| Phase 3 | 動的トポロジ（分割/統合/接続） | 機構は動作；現状は固定トポロジに劣る |
| Phase 4 | 学習型構造自己修正 | ポリシーが自己状態から構造決定を学習（下記参照） |
| Phase 5 | 内在的パラメータ自己修正 | 閉ループ `θ→h→Δθ→θ'` が存在・安定・状態依存的（下記参照） |

### Phase 1 — 継続学習（Control / Exp1 / Exp2）

| 指標 | Control | Exp1 | Exp2 |
|---|---|---|---|
| 平均忘却 AF ↓ | 0.0037 | **0.0010** | 0.0013 |
| 前方転移 FWT ↑ | 0.0002 | −0.0041 | **+0.0013** |
| 継続学習スコア CLS ↑ | **0.0868** | 0.0735 | 0.0565 |
| 平均獲得 ↑ | 0.0942 | 0.0725 | 0.0548 |
| 平均保持 ↑ | **0.0905** | 0.0745 | 0.0578 |

*Control = 逐次ファインチューニング；Exp1 = 単一ネットワーク＋選択的内化；
Exp2 = 5 ネットワーク＋ネットワーク間検証。*

![Phase 1 曲線](experiments/phase1_comparison_curves.png)
![Phase 1 指標](experiments/phase1_metrics.png)

### Phase 2 — 能動学習

| 戦略 | 最終性能 | 最終カバレッジ |
|---|---|---|
| random（ベースライン） | 0.0572 | 0.256 |
| uncertainty | 0.0466 | 0.256 |
| **info_gain** | **0.0591** | 0.256 |
| meta | 0.0561 | 0.256 |

![Phase 2 曲線](experiments/phase2_curves.png)

### Phase 3 — 構造進化

| 指標 | fixed | evolve |
|---|---|---|
| 最終平均性能（5 ドメイン） | **0.0575** | 0.0533 |
| コードドメイン適応（round 4→7 Δ） | **+0.0130** | +0.0041 |
| 最終ネットワーク数 | 5 | 5 |

観測された進化イベント：統合（round 7）、分割（round 9）、分割（round 11）、
統合＋動的接続（round 13）。

![Phase 3 コード曲線](experiments/phase3_code_curve.png)

### Phase 4 — 学習型構造自己修正（rule vs learned vs fixed）

分布シフト流（general(4)→code(4)→mixed_code(4)→science(4)、16 ラウンド）で
3 つの制御方式を比較。`rule` と `learned` は同じ「候補→評価→受理/ロールバック」
機構を使い、唯一の違いは構造アクションを**誰が決定するか**（人手ルール vs
学習済みポリシー）。

| 指標 | fixed | rule | learned |
|---|---|---|---|
| 最終平均性能（5 ドメイン） | 0.0554 | 0.0570 | **0.0585** |
| 平均忘却 AF ↓ | **0.0000** | 0.0099 | 0.0029 |
| 前方転移 FWT ↑ | 0.0005 | **0.0009** | 0.0007 |
| 継続学習スコア CLS ↑ | 0.0553 | 0.0471 | **0.0556** |
| コードドメイン適応（r4→8 Δ） | +0.0224 | +0.0326 | **+0.0336** |
| 最終ネットワーク数 | 5 | 2 | 2 |
| 修正成功率 | — | 1.00 | 1.00 |
| 修正平均報酬 | — | — | −0.0031 |

この単一シードの実行では、learned 制御方式が最高の最終平均性能・最高 CLS・
最良のコードドメイン適応を示し、rule 制御方式より忘却が小さくなりました。
learned ポリシーは Stage B で構造修正を**自律的に提案**し（r11 で統合）、
アクション・エントロピーが学習とともに低下しました（挙動の変化）。
**注意：** 単一シード・プロトタイプ規模であり、差（≈0.003）は統計的に
確立されていません。報酬信号は僅かに負で、ポリシーは生のルールエンジンより
保守的になりました。

設計と議論の詳細は [docs/PHASE4.md](docs/PHASE4.md)。

![Phase 4 比較](experiments/phase4/phase4_comparison.png)
![Phase 4 アクション分布](experiments/phase4/phase4_actions.png)
![Phase 4 報酬](experiments/phase4/phase4_reward.png)
![Phase 4 学習曲線](experiments/phase4/phase4_learning.png)

### Phase 5 — 内在的パラメータ自己修正（θ → h → Δθ → θ'）

各ネットワークは `IntrinsicPlasticityModule`（ネットワークの**メンバ**）を
持ち、自身の内部状態を連続的なパラメータ変化に写像します。P5 の中心的命題は
「パラメータ―状態フィードバック閉ループ」の存在性・安定性・状態依存性・
識別可能性であり、**性能は記述的指標であり証明には用いません**（設計
レポートに従う）。20 ラウンドのシフトストリーム
（general(5)→code(5)→mixed_code(5)→science(5)）：

| 指標 | fixed | p5b（内在） | random | constant | shuffled |
|---|---|---|---|---|---|
| 最終平均性能（5 ドメイン） | **0.0413** | 0.0412 | 0.0405 | 0.0409 | 0.0407 |
| 平均忘却 AF ↓ | **0.0090** | 0.0090 | 0.0097 | 0.0094 | 0.0095 |
| 継続学習スコア CLS ↑ | **0.0323** | 0.0322 | 0.0308 | 0.0315 | 0.0312 |
| トリガ数 / 受容率 | — | 100 / 1.00 | 100 / 1.00 | 100 / 1.00 | 100 / 1.00 |
| Δθ 平均ノルム | — | **1.285** | 1.310 | 1.325 | 1.288 |
| Δθ ノルム分散 | — | **4.8e-3** | 1.3e-2 | 3.3e-3 | 1.8e-3 |
| 予測変化率 | — | **0.014%** | 0.019% | 0.012% | 0.007% |

中核検証（単一シード、全て合格）：Δθ は非ゼロ（‖ΔW_A‖≈0.67、
‖ΔW_B‖≈0.69）、状態依存的（同一入力なら決定論的；入力間差 ≈0.23）、
パラメータを遷移（‖θ'−θ‖≈32.7）、挙動を変更（logits 差 ≈0.21）、
閉ループは非定常かつ発散せず、20 ステップ安定（NaN なし、エントロピー
≈3.87）；ランダム/固定/シャッフル Δθ と識別可能。同一ノルムの Δθ のうち、
内在 Δθ はイベント間分散が**最小**・挙動撹乱が**最小**——状態に整合した
構造化修正と整合します。

**P5-C**（オフライン適応的可塑性学習）は [docs/PHASE5.md](docs/PHASE5.md)
§5.4 に別途報告：機構は完全に動作（100 トリガ、各ネットワーク 19/20 成功
ケースと 11 回の訓練、平均報酬 +9.6e-4）しましたが、報酬重み付きオフライン
信号（≈1e-9）は本規模では無視でき、測定可能な可塑性改善は観測されません
でした。より高い最終指標は追加の適応計算による交絡です。

![Phase 5 性能](experiments/phase5/phase5_perf.png)
![Phase 5 閉ループ](experiments/phase5/phase5_loop.png)
![Phase 5 対照](experiments/phase5/phase5_controls.png)
![Phase 5 学習](experiments/phase5/phase5_learning.png)

## 現在の知見

以下の知見は**暫定的**であり、本プロトタイプとその実験設定に限定されます。
継続学習やニューラルアーキテクチャ設計に関する一般的結論として解釈すべき
ではありません。

1. **知見 1** — 固定計算予算の下では、マルチネットワーク協調は絶対性能を
   自動的に向上させない。
2. **知見 2** — 漸進的内化は、現プロトタイプにおいて破壊的忘却の低減に
   *有望*（AF：0.0037 → 0.0010 / 0.0013；ドメイン別忘却も一貫して低い）。
3. **知見 3** — 情報利得に基づく経験選択は、テストした設定でランダム選択より
   *有望な結果*を示す。
4. **知見 4** — 動的トポロジ進化は現状、固定トポロジを上回るには*不十分*で、
   より良い構造可塑性制御が必要（分割/統合中の短期的性能擾乱を観測——
   設計レポートのリスク分析と整合）。
5. **知見 5（Phase 4）** — ルール模倣＋REINFORCE で訓練した小さなポリシーが、
   ルールエンジンなしでシステム自身の状態から構造修正決定を*生成できる*こと
   を示した。ルールや固定トポロジを*上回るか*は、このプロトタイプ規模では
   未確立（[docs/PHASE4.md](docs/PHASE4.md) と `experiments/phase4` 参照）。
6. **知見 6（Phase 5）** — モデルの内部状態は、自身のパラメータへの状態依存的・
   安定的・測定可能な変更を*直接生成できる*（`θ → h → Δθ → θ'`）。閉ループは
   Test 1-6 を通過し、ランダム/固定/シャッフル修正と区別可能——性能向上は
   主張しません（[docs/PHASE5.md](docs/PHASE5.md) と `experiments/phase5` 参照）。

## 再現

環境：Python 3.8.16、PyTorch 1.13.1、CUDA 11.7
（`transformers==4.45.2`、`datasets==2.21.0`、`peft==0.12.0`、
`accelerate==0.34.2`、`huggingface_hub==0.25.2`、`tokenizers==0.20.1`；
`requirements.txt` 参照）。元実験は NVIDIA RTX 3070 Ti 8GB で実施。

```bash
# 1) ベースモデル（GPT-2 small）をダウンロード — 約 550 MB
python scripts/download_model.py

# 2) データセット準備（data/ にダウンロード・キャッシュ）
python -c "from scripts.common import make_config, prepare_data; \
           d = prepare_data(make_config()); print({k: len(v) for k, v in d['train'].items()})"

# 3) Phase 1 — 継続学習（Control / Exp1 / Exp2）
python scripts/run_phase1.py --modes control exp1 exp2 --out experiments/phase1

# 4) Phase 2 — 能動学習
python scripts/run_phase2.py --out experiments/phase2

# 5) Phase 3 — 構造進化
python scripts/run_phase3.py --out experiments/phase3

# 6) Phase 4 — 学習型構造自己修正（rule vs learned vs fixed）
python scripts/run_phase4.py --out experiments/phase4

# 7) Phase 5 — 内在的パラメータ自己修正
#    (a) 中核検証：Test 1-6 ＋ 負の対照
python scripts/validate_phase5.py
#    (b) 主実験：fixed vs intrinsic (p5b) — 20 ラウンド
python scripts/run_phase5_b.py --out experiments/phase5
#    (c) 負の対照アーム：ランダム / 固定 / シャッフル Δθ
python scripts/run_negative_controls.py --out experiments/phase5
#    (d) P5-C：適応的可塑性学習
python scripts/run_phase5_c.py --out experiments/phase5
#    (e) 分析テーブルと図
python scripts/analyze_phase5.py --out experiments/phase5

# 8) 結果を experiments/comparison.md と図に集約
python scripts/make_report.py
```

ネットワーク制限の注意：HuggingFace Hub や PyPI に到達できない環境向けに、
再開可能なリゾルバ/ダウンローダ（`scripts/resolve_deps.py` +
`scripts/download_wheel.py`）を同梱しています。元環境もローカルプロキシ経由で
この方式によりブートストラップしました。

## プロジェクト構造

```
dscns/
├── dscns/                  # 中核実装（19 モジュール、Phase 4+5 含む）
├── scripts/                # ダウンロード / 実験 / レポートスクリプト
├── config/                 # phase1.yaml, phase5.yaml
├── tests/                  # Phase 5 検証スイート（Test 1-6 ＋ 負の対照）
├── docs/                   # 設計・実験・限界・ライセンス・PHASE4・PHASE5
├── experiments/            # 公式結果（JSON + 図）— Git に保持
├── REPORT_zh.md            # 中国語再現レポート
├── requirements.txt
├── LICENSE                 # GPL-3.0（コード）
└── README(.zh-CN/.ja-JP).md
```

`models/`・`data/`・`wheelhouse/` はローカルキャッシュであり、`.gitignore`
によりバージョン管理から除外されています。

## 設計ノート

- 実装は DSCNS 設計レポート v1.0 に従います（英語の設計要約は
  `docs/DESIGN.md`）。
- Phase 4 は設計レポート修正案《DSCNS 自主神経構造自修正メカニズム》に従い
  （`docs/PHASE4.md`）、構造修正の*決定*をポリシー
  （`SelfModificationPolicy`、模倣＋REINFORCE）が学習し、
  `StructureEvolver` は実行とハード安全制約を担います。
- Phase 5 は設計レポート《DSCNS Phase 5 内生式参数自修改机制》に従い
  （`docs/PHASE5.md`）、ネットワーク内部状態が直接、連続的なパラメータ変化を
  生成します（`IntrinsicPlasticityModule`、`θ → h → Δθ → θ'`）。トリガ・
  検証・ロールバックは実験コントローラの責務です。
- マルチネットワーク＝共有凍結ベース＋ネットワーク別 LoRA アダプタ。
  パラメータ空間 Θ_i は独立、記憶は共有。
- 知識項目ごとの状態レベルと内化度 I_ij ∈ [0,1] を記録し追跡可能にします。
- 設計レポートからの既知の逸脱は `docs/EXPERIMENTS.md`（§7）と
  `docs/LIMITATIONS.md` に記載。

## 限界

- モデル規模が小さい（124M ベース）うえ、データ予算も限定的。
- マルチネットワーク実験は厳格な計算予算の対等性に拘束。
- 構造進化の機構は意図的にシンプル。Phase 4 の学習型制御は小さなポリシーと
  極小の RL 予算（概念実証であり、スケーラブルなアーキテクチャ探索ではない）。
- Phase 5 の内在的可塑性はプロトタイプ規模でのみ検証（単一シード、GPT-2
  small、単一 LoRA ランク、外部固定トリガ）；性能向上は主張しません。
- 大規模ベンチマーク未実施；他モデル・他領域への一般化の証拠はまだない。
- DSCNS が成熟した継続学習手法（EWC、経験リプレイ等）を一般に上回るという
  証拠はまだない。
- 完全なリストは `docs/LIMITATIONS.md` を参照。

## 今後の課題

- より長い学習期間と大きな予算で仮説 H1/H2 を再検証。
- 手調整ではなく学習される関連性・信頼重み関数。
- Phase 4：より大きな RL 予算、より長い適応ウィンドウ、レイヤー単位
  （アダプタ集団単位に留まらない）の学習型自己修正。
- Phase 5：エンドツーエンド微分可能な可塑性、学習型トリガ（Level 4）、
  完全パラメータ条件付け、多層連携；次段階は Phase 6（内在的*構造*自己修正）。
- より良い構造可塑性制御（適応的進化閾値、進化後の安定化スケジュール）。
- より大規模なベンチマーク（EWC / リプレイ / PEFT ベースライン）。
- マルチモーダル・オープン環境への拡張（設計レポート Phase 6+）。

## 引用

本リポジトリを研究で利用する場合は、以下を引用してください：

```bibtex
@misc{dscns2026,
  title  = {DSCNS: Dynamic Self-Modifying Cognitive Network System -- A Research Prototype},
  author = {Mousennnn},
  year   = {2026},
  month  = {aug},
  note   = {Version v0.3.0, early research prototype},
  howpublished = {GitHub repository},
  url    = {https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System}
}
```

## ライセンス

- **コード：** [GNU General Public License v3.0](LICENSE)
- **設計ドキュメント・実験レポート**（`docs/`、`REPORT_zh.md`、
  `experiments/comparison.md`）：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  — `docs/LICENSE-docs.md` 参照。

**帰属要件：** ドキュメントを再利用・改変する際は、以下をクレジットして
ください：

> DSCNS — Dynamic Self-Modifying Cognitive Network System (v0.3.0)、
> Mousennnn 著、CC BY 4.0 ライセンス。
> https://github.com/Mousennnn/DSCNS-Dynamic-Self-Modifying-Cognitive-Neural-System

---

*これは初期研究プロトタイプです。実験結果は否定的・混合的な知見も含め、
正直に報告されています。*
