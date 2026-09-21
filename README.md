# Busbar

<p align="center">
  <a href="README.md"><kbd>简体中文</kbd></a> &nbsp; <a href="README.en.md"><kbd>English</kbd></a>
</p>

**让应用基于同一份上下文，反复完成结构化判断。**

Busbar 是一个可在 Apple Silicon Mac 上运行的开源 LLM 决策运行时。应用提交上下文后，Busbar 将它编译成可复用的 `ContextSnapshot`；后续问题复用模型已经算过的前缀，返回布尔判断、单选结果或带分布的数值评分。

例如，一份客服工单可以同时用于判断“是否重复扣款”、选择“交给哪个团队”、评估“处理优先级”。应用负责定义问题和候选描述、执行后续业务动作，Busbar 负责模型计算、上下文复用和结构化结果。它适合工单分流、规则判断、候选筛选等有限选项任务；需要长文本创作或自由对话时，仍应使用生成接口。

```text
业务上下文 → 分词与一次 prefill → ContextSnapshot
                                  ├─ Boolean：是否需要退款？
                                  ├─ Choice：交给哪个团队？
                                  └─ Score：处理优先级是多少？
```

## 两种接入方式

| | 快捷模式 | 可训练选择头 |
|---|---|---|
| 如何得到分数 | 读取已有词表 A/B/C/D 等标签的分数 | 每个候选描述经过底座，再由共享评分层打分 |
| 是否需要训练 | 不需要 | 需要标注数据；冻结底座，只训练小头 |
| 适合什么情况 | 现有指令模型已经能理解任务，先快速接入 | 有任务数据，需要学习自己的判断规则或偏好 |
| 支持后端 | MLX、HF/PyTorch、vLLM Metal | 原生 MLX |
| 应用接口 | 共用 Boolean / Choice / Score 与 snapshot 生命周期 | 同左 |

原生 MLX/HF 可以只计算标签对应的词表行。选择头采用 NanoJev 风格的候选描述评分：`zᵢ = wᵀ LayerNorm(hᵢ)`，再在候选之间做 softmax。两条路线都复用上下文，但选择头有 K 个候选就要运行 K 条后缀路径，**不能仅凭输出层更小就认定更快或更准**。编码、训练及取舍见 [选择头指南](docs/trainable-heads.md)。

## 与官方 Jev 的跑分对比（部分题集）

2026-09-21，我们在 [Typed Decision Bench v0.3](https://blobfish.ai/benchmarks/typed-decision-bench)（TDB）的**同一批 4,370 条样本**上，对比 Busbar 与官方托管模型 `jev-1.13.0`。TDB 是第三方公开评测，不是 TypeSafe 官方私有四工作流评测；Jev 一列来自上游公开的逐题作答，本次没有重新调用 Jev API。

Busbar 使用 **Qwen3.5-4B、BF16、原生 MLX、快捷词表模式**，运行在 M4 Pro / 48 GiB Mac 上。**本表不包含可训练选择头或正在进行的训练优化结果。** 两边均使用上游评分函数，并在同一个样本 ID 子集上重新计算。

| Suite | 样本数 | Jev DS ↑ | Busbar DS ↑ | Jev 准确率 | Busbar 准确率 |
|---|---:|---:|---:|---:|---:|
| data-and-operations | 722 | 82.75 | 78.40 | 78.67% | 77.42% |
| real-time-and-agents | 1,199 | 75.81 | 70.47 | 65.47% | 58.22% |
| retrieval-and-knowledge | 890 | 82.49 | 77.54 | 79.33% | 71.12% |
| safety-and-quality | 1,162 | 88.91 | 85.83 | 85.89% | 80.72% |
| workflow-control | 397 | 85.27 | 81.63 | 81.86% | 76.07% |
| **同题子集合计** | **4,370** | **82.66** | **78.32** | **77.39%** | **71.62%** |

DS 指 DecisionScore（0–100，越高越好），按每个样本的 `1 − 归一化 Brier` 聚合，衡量概率预测质量；不是准确率。此子集上，Busbar 比 Jev 低 **4.34 分 DS**、**5.77 个百分点准确率**。

**覆盖范围：我们只跑了部分样本，不能视为全量榜单成绩。**

- TDB 共 5,387 条样本；其中 359 条只公开 ID、未公开正文，排除后有 5,028 条可运行样本。
- Busbar 本次回答 4,370 条：占全量 **81.12%**，占公开正文样本 **86.91%**。另外 658 条含 28–64 个候选，超过本次运行的 26 候选上限，因此排除。
- Jev 也只在这 4,370 条上计分。上表不能与 TDB 全量榜单直接排位，也不能把差距单独归因于架构、训练方法或推理后端。
- Busbar 本机推理与 Jev 托管 API 的计时边界不同，不据此给出速度倍数。

可核对 [评分汇总 JSON](benchmarks/results/tdb-2026-09-21/compare-4370.json)（含逐 suite / task 指标、上游评分器版本及结果文件 SHA256）、[运行配置](benchmarks/results/tdb-2026-09-21/busbar-qwen35-full.manifest.json)和[详细对照与复现说明](docs/jev-comparison.md)。题目与 Jev 作答来自 [TDB 公开数据集](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench)；该目录仅发布汇总与配置，不包含题目正文或逐题原始作答。

## 在 Mac 上开始

需要 Apple Silicon、macOS 15+、Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/CCpro10/Busbar.git
cd Busbar
uv sync --locked --extra mlx
uv run busbar run examples/returns.json
```

默认加载固定 revision 的 Qwen3-0.6B，首次下载约 1.2 GB 权重；之后模型和输入均在本机运行。返回结果包含每个候选的分数、概率、最终取值，以及实际计算/复用 token 数和耗时。较大模型已在 M4 Pro、48 GiB Mac 上验证：

```bash
uv run busbar run examples/returns.json --model openbmb/MiniCPM5-2B --dtype bfloat16
uv run busbar run examples/returns.json --model Qwen/Qwen3.5-4B --dtype bfloat16
```

MiniCPM5 实际约 2.52B；Qwen3.5 加载官方 checkpoint 的文本部分。它们均未量化，内存需求高于默认 0.6B。安装、下载体积、HF 参考路径、vLLM Metal 独立环境与已知限制见 [后端指南](docs/backends.md)。

## 接入 Python 或 HTTP

```python
from busbar import ContextSpec, DecisionRequest, Runtime
from busbar.backends import load_backend

runtime = Runtime(load_backend("mlx"))
context = runtime.compile_context(
    ContextSpec(namespace="ticket-1", state={"message": "我只付款一次，银行卡却扣了两次"})
)
result = runtime.decide(
    DecisionRequest(
        namespace="ticket-1",
        snapshot_id=context.snapshot.id,
        questions={
            "billing": {
                "type": "boolean",
                "question": "这条诉求是否需要支付或账单团队处理？",
            }
        },
    )
)
print(result.decisions["billing"].model_dump())
# 后续问题继续使用 context.snapshot.id；结束后可主动释放。
runtime.delete_context(context.snapshot.id, namespace="ticket-1")
```

需要跨 HTTP 请求保留模型与缓存时，运行 `uv run busbar serve --port 8787`，打开 [本机接口文档](http://127.0.0.1:8787/docs)。先 `POST /v1/contexts`，再用返回的 ID 调用 `POST /v1/decisions`；一次可混合 1–64 道题。上下文支持创建、复用、版本化、列表/筛选、删除和 namespace 清理。

Choice 支持 2–26 个带 ID 与描述的选项；Score 支持 2–26 个带描述的递增数值等级，返回概率加权期望。概率是候选集合内的归一化分数，未经业务校准；熵也不代表正确率。[接口指南](docs/api.md) 解释完整字段、生命周期、错误码与内存约束。

## 训练自己的选择头

```bash
uv run busbar train-head examples/trainable_support/train.jsonl \
  --validation examples/trainable_support/validation.jsonl \
  --output models/heads/support-v1 --epochs 40

uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --head models/heads/support-v1 --output local-results/support-v1-test.json

uv run busbar serve --head models/heads/support-v1 --port 8787
```

每个版本保存小头权重、训练记录、数据划分和文件哈希，不复制底座。支持从旧头继续训练、加载校验与版本回退；新版本使用新目录，已有版本不覆盖。应用只需替换 `load_backend("mlx", head="models/heads/support-v1")`，决策 API 保持一致。数据格式、验证集选参、继承链防泄漏及失败处理见 [完整训练指南](docs/trainable-heads.md)。

## 证据与当前边界

仓库包含普通回归、实际模型数值测试、真实 HTTP 验证，以及 SemIf、NanoJev 和 Busbar 的对照原始结果。[与官方 Jev 的同题对照](docs/jev-comparison.md) 在 Typed Decision Bench 的 4,370 道公开题上用上游评分函数比较 Busbar 与托管 `jev-1.13.0`。[验证指南](docs/validation.md) 区分实现正确性、任务质量、端到端延迟与输出头微测量；[0.3](docs/mac-v03.md) 和 [0.4](docs/mac-v04.md) 给出实际设置、逐题结果、比较和失败记录。

当前服务面向本机：一个进程驻留一个模型，模型操作串行，请求内部可以分批；KV 在内存中，重启后重建。namespace 不提供认证。MiniCPM5/vLLM Metal 的缓存数值一致性尚未通过已有门禁；首次接入可使用原生 MLX。Linux 服务引擎、连续批处理、量化和底座微调尚未实现。

本项目是独立实现，借鉴 Jev 类结构化决策思路；不是 TypeSafe Jev 的官方实现，也不是 NanoJev checkpoint 兼容加载器。底层依赖 [MLX-LM](https://github.com/ml-explore/mlx-lm)、[Transformers](https://github.com/huggingface/transformers) 及所选模型。

## 阅读与参与

- [接口与运行时](docs/api.md)：应用接入、完整生命周期及错误处理。
- [后端与模型](docs/backends.md)：安装、平台差异和支持范围。
- [可训练选择头](docs/trainable-heads.md)：评分原理、数据、训练、评测和回退。
- [架构与扩展](docs/design.md)：模块职责、数据流、不变量和后端接入契约。
- [开发指南](CONTRIBUTING.md)：环境、代码规范、测试与提交要求。
- [变更记录](CHANGELOG.md)：版本变化；[0.4.1 整理记录](docs/quality-v041.md) 说明本次修复与验证。

许可证：[MIT](LICENSE)。
