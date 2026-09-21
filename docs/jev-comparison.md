# 与官方 Jev 在 Typed Decision Bench 上的同题对照

2026-09-21，Apple M4 Pro，48 GiB，macOS 15.6.1。本机推理，未调用 Jev API。

这次回答一个此前无法回答的问题：**把 Busbar 和真正的官方 Jev 放在同一批题上，差多少。**
TypeSafe 官方明确不发布公开基准成绩，其四工作流评测只公布分数、不公开题目，因此无法复跑。
[Typed Decision Bench](https://blobfish.ai/benchmarks/typed-decision-bench)（下称 TDB）提供了另一条路：
题目、ground truth 和**托管 Jev `jev-1.13.0` 的逐题作答**全部公开，所以可以用它的题、它的
评分函数，把两边放进同一张表。

## 数据与评分口径

- 题目与官方作答来自 [TDB v0.3 数据集](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench)，
  5,387 题、25 任务、5 suite；标签来自 21 个公开数据集，非模型判定。
- 评分函数直接从上游仓库 `80262b5c7049179a5def5e15fc85c51e3f9d7314` 导入，不重写：
  DecisionScore = 每题 `1 − 归一化 Brier` 的均值 ×100，noul 用 `(p−y)²`，choice/score 用
  `Σ(p−y)²/2`。
- 359 题（`abt-buy-product-match`、`trialgpt-criterion-support`）以 ids-only 发布，正文不可再分发，
  两边都无法作答，排除；其余 5,028 题上复算官方 Jev 得到 DecisionScore 81.43、准确率 76.57%，
  四个未受影响 suite 为 83.92 / 75.81 / 88.91 / 74.70，与榜单公布值逐项吻合——这是评分路径可信的依据。

**本次对照的分母是 4,370 题**：Busbar 的候选上限为 26，TDB 有 658 题的最宽问题达到 28–64 个选项，
Busbar 无法表示。这些题两边都不计入，官方 Jev 也在同一批 4,370 题上重新评分；把 Jev 在超限题上的
成绩算进来、再和 Busbar 的 4,370 题相比，是不同题集戴同一个表头。

## 结果

同一批 4,370 题，同一评分函数。Busbar 使用 Qwen3.5-4B、BF16、快捷模式（词表读出）。

| suite | 题数 | Jev DS | Busbar DS | 差 | Jev 准确率 | Busbar 准确率 |
|---|---:|---:|---:|---:|---:|---:|
| data-and-operations | 722 | 82.75 | 78.40 | −4.35 | 78.7% | 77.4% |
| real-time-and-agents | 1199 | 75.81 | 70.47 | −5.34 | 65.5% | 58.2% |
| retrieval-and-knowledge | 890 | 82.49 | 77.54 | −4.95 | 79.3% | 71.1% |
| safety-and-quality | 1162 | 88.91 | 85.83 | −3.08 | 85.9% | 80.7% |
| workflow-control | 397 | 85.27 | 81.63 | −3.64 | 81.9% | 76.1% |
| **合计** | **4370** | **82.66** | **78.32** | **−4.34** | **77.4%** | **71.6%** |

**Busbar 落后 4.34 个 DecisionScore、5.8 个百分点准确率。** 作为参照，TDB 榜单上 Jev 与第二名
JevFish 相差 3.46，与 decider-2b 相差 4.14；Busbar 这次的差距与该量级相当，但这不是排名，
两者的底座、训练和运行环境都不同。

差距在 5 个 suite 上方向一致，说明不是某类任务的偶发问题。准确率的差距（−5.8）明显大于
DecisionScore 的差距（−4.34），意味着 Busbar 选错的题更多，但它给错误答案的概率相对不那么极端。

### 任务级分布

20 个任务中 Busbar 领先 4 个、落后 16 个。

| 任务 | Jev | Busbar | 差 | 题数 |
|---|---:|---:|---:|---:|
| swe-smith-run-resolved | 60.20 | 69.78 | **+9.58** | 200 |
| cuad-checklist | 91.61 | 93.71 | +2.10 | 204 |
| unfair-tos-clause-type | 85.91 | 87.51 | +1.60 | 200 |
| prompt-injection | 90.54 | 91.41 | +0.87 | 194 |
| …（中间 12 项，−1.80 至 −7.86） | | | | |
| tau2-airline-run-success | 68.02 | 59.34 | −8.68 | 200 |
| vitaminc-claim-consistency | 88.08 | 79.24 | −8.84 | 200 |
| hotpot-context-filter | 96.30 | 87.24 | −9.06 | 200 |
| tau2-airline-transcript-success | 76.74 | 57.45 | **−19.29** | 199 |

值得注意的是 `swe-smith-run-resolved`：Busbar 反超 9.58，而这是 Jev 全场表现最差的任务之一
（60.20）。差距最大的 `tau2-airline-transcript-success`（−19.29）是长对话记录判定，与
`real-time-and-agents` 整体偏弱一致——长上下文是当前较明显的短板。单任务约 200 题，
上游明确说明 3 分以内的差异属于噪声，因此只有绝对值较大的几项值得追。

### 延迟不可直接比较

| | 中位 | p95 |
|---|---:|---:|
| Busbar（本机 M4 Pro，MLX，4B） | 581 ms | 2258 ms |
| Jev（托管 API，从上游记录读取） | 182 ms | 260 ms |

两者测量边界不同：Busbar 是本机端到端（含每题一次 state prefill），Jev 是上游在其机器上记录的
托管 API 往返。上游也明确标注延迟不参与排名。这组数字只说明"本机 4B 能在亚秒内完成一题"，
不构成"Jev 快 3.2 倍"的结论。

## 这次放开了候选上限

原上限 16 来自标签字母表 `ABCDEFGHIJKLMNOP`，而非后端限制——投影本身可以取任意多行标签。
本次扩到完整 A–Z，上限 26，提示词格式不变，因此与 0.3 / 0.4 报告仍可对比。

上限由实测决定，而非可分词性。在 TDB 的 26 选项任务上（60 题，Qwen3.5-4B，BF16）对照了四套标签：

| 标签方案 | 26 选项准确率 | 64 选项 |
|---|---:|---|
| A–Z | **96.67%** | 不可用（仅 26 个） |
| 大小写字母 | 96.67% | 不可用（仅 52 个单 token） |
| 两字母组 AA/AB… | 96.67% | 78.33% |
| 希腊+西里尔 | 95.00% | 66.67% |
| **单 token 特殊字符** | **41.67%** | 不可用（仅 62 个单 token） |

特殊字符方案在分词层面完全合格——单 token、不破坏边界、ID 不冲突——但准确率掉 55 个百分点：
**能切成一个 token，不等于模型会把它当成选项标识。** 该方案已排除。

继续放开到 64 需要两字母组，但它在 64 选项上降到 78.33%，且会改变提示词格式、使历史结果不可比，
因此本次停在 26，未实现 Jev 文档所述的 255 选项。当前仍有 658 题（4 个任务）因此无法作答。
选择头路线按候选描述编码、不依赖字母表，不受该上限约束，但本次未用它跑 TDB。

## 不能由此得出的结论

- **不是架构或训练方法的比较。** Jev 是闭源托管模型，训练数据未公开，TDB 将其全部污染层级标记为
  unknown；Busbar 用的是公开 Qwen3.5-4B。差距里有底座、训练、规模、服务栈多个变量，无法归因到单一原因。
- **不是官方四工作流评测的复现。** 那套题目不公开，本文没有复跑，也没有把 TDB 结果填进官方口径。
- **不能与 TDB 榜单并列。** 榜单是 5,387 题全量口径，本文是 4,370 题子集，分母不同。
- **概率未经业务校准。** DecisionScore 衡量概率质量，但 Busbar 返回的是候选集合内的条件概率，
  低 Brier 不代表已校准。

## 复现

```bash
# 1. 取题目与官方作答（bench/SHA256SUMS 可校验；retrieval 一项因 withheld 正文会失配）
#    数据集：SamuelChien821/typed-decision-bench

# 2. Busbar 作答同一批题
uv run python benchmarks/run_tdb.py --bench local-results/tdb/bench \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --label busbar-qwen35-4b --output local-results/tdb/busbar-qwen35-full.jsonl

# 3. 用上游评分函数在共同题目上评分两边
uv run python benchmarks/tdb_score.py --bench local-results/tdb/bench \
  --source-root <clean jevfish checkout> \
  --only-ids local-results/tdb/answered-ids.json \
  --result "jev-1.13.0=local-results/tdb/results/jev-1.13.0.jsonl" \
  --result "busbar-qwen35-4b=local-results/tdb/busbar-qwen35-full.jsonl" \
  --output local-results/tdb/compare-4370.json

# 4. 标签方案对照
uv run python benchmarks/label_scheme_probe.py \
  --items local-results/tdb/bench/suites/data-and-operations/items.jsonl \
  --task dbpedia-ontology-hierarchy --limit 60 \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --output <path>
```

全量运行 3644 秒，4370/5028 作答，结果文件 SHA256
`a1978a6e9d15bc7f11144d6dc887867ec2535deae16f098fe4e4a5a91766c24d`。
逐题作答、概率与耗时保留在 `local-results/tdb/`（不入库）。
