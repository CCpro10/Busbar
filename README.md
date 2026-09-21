# Busbar

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

Choice 支持 2–16 个带 ID 与描述的选项；Score 支持 2–16 个带描述的递增数值等级，返回概率加权期望。概率是候选集合内的归一化分数，未经业务校准；熵也不代表正确率。[接口指南](docs/api.md) 解释完整字段、生命周期、错误码与内存约束。

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
