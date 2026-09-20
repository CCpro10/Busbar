# Busbar

A context-reuse decision runtime for LLM agents.

Busbar 把上下文编译成可复用的 `ContextSnapshot`：同一份 state 只做一次 prefill，后续 Boolean、Choice、Score 判断从独立 KV 分支计算问题后缀。第一版面向 Apple Silicon Mac，使用 Qwen3-0.6B 和 MLX；HF/PyTorch 提供独立的 CPU 正确性参考。

M4 Pro 本机实测：4096-token 前缀、16 题，fresh 逐题中位 13.72 秒；暖缓存批量中位 0.86 秒；首次编译加批量判断 1.61 秒。3 次测量存在波动，小模型也出现判断错误；完整条件、分数差异及原始数据见 [验收记录](docs/acceptance.md)。

```text
Context JSON → 稳定序列化与分词 → 一次 prefill → ContextSnapshot
                                                   ├── question 1 → decision
                                                   ├── question 2 → decision
                                                   └── question N → decision
```

## 在 Mac 上运行

需要 Apple Silicon、macOS 15+、Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。首次运行会从 Hugging Face 下载约 1.2 GB 的模型权重；权重保存在用户的 Hugging Face 缓存中，不进入本仓库。

```bash
git clone https://github.com/CCpro10/Busbar.git
cd Busbar
uv sync --extra mlx --extra dev
uv run busbar run examples/returns.json
```

默认固定模型 `Qwen/Qwen3-0.6B`、revision `c1899de289a04d12100db370d81485cdf75e47ca`，MLX 使用 FP16，关闭额外思考与文本生成。输入、模型权重和推理均在本机；下载模型需要网络。

## Python：先编译，再反复判断

```python
from busbar import Runtime, ContextSpec, DecisionRequest
from busbar.backends import load_backend

runtime = Runtime(load_backend("mlx"))
context = runtime.compile_context(
    ContextSpec(
        namespace="session-1",
        state={"delivered_days_ago": 7, "unused": True},
    )
)
answer = runtime.decide(
    DecisionRequest(
        namespace="session-1",
        snapshot_id=context.snapshot.id,
        questions={
            "unused": {
                "type": "boolean",
                "question": "Is the item unused?",
            }
        },
    )
)
print(answer.model_dump_json(indent=2))
runtime.delete_context(context.snapshot.id, namespace="session-1")
```

相同 namespace、模型版本、指令和 JSON 内容会复用同一个 snapshot；JSON 对象的字段顺序不影响身份。问题不会写回 context。更新 state 时创建新版本，旧版本在未过期或被淘汰时仍可用于回退。删除后再调用会明确报错，重新提交原 state 即可重建。

## HTTP：让模型和缓存跨请求驻留

```bash
uv run busbar serve --port 8787
```

服务只监听 `127.0.0.1`，一个进程、一个模型。交互式接口文档在 <http://127.0.0.1:8787/docs>。

```bash
curl http://127.0.0.1:8787/v1/contexts \
  -H 'Content-Type: application/json' \
  -d '{"state":{"unused":true},"namespace":"demo"}'

# 将响应 snapshot.id 填入下方；之后只需发送 snapshot ID 和新问题。
curl http://127.0.0.1:8787/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"namespace":"demo","snapshot_id":"<snapshot.id>","questions":{"unused":{"type":"boolean","question":"Is the item unused?"}}}'
```

| 接口 | 行为 |
|---|---|
| `POST /v1/contexts` | 创建或复用上下文 |
| `GET /v1/contexts?namespace=demo&id_prefix=...` | 列表及 ID 前缀过滤 |
| `GET /v1/contexts/{id}?namespace=demo` | 查询元数据，不延长 TTL |
| `PUT /v1/contexts/{id}` | 以新内容创建新版本，返回新 ID |
| `DELETE /v1/contexts/{id}?namespace=demo` | 幂等删除 |
| `DELETE /v1/contexts?namespace=demo` | 清空一个 namespace |
| `POST /v1/decisions` | 一次提交 1–64 道题 |
| `GET /v1/stats`、`GET /health` | 缓存占用、命中、淘汰及模型状态 |

## 决策语义

- **Boolean**：`type="boolean"`，返回 `true/false` 的概率，`value` 是 `P(true) >= 0.5`。
- **Choice**：2–16 个互斥选项，每项具有唯一 `id` 和 `description`；`value` 是最大分数选项的 ID。
- **Score**：2–16 个具有明确描述、严格递增的数值等级；`value = Σ(probability × level.value)`，同时保留完整分布。

所有结果包含原始 logits、候选内 softmax 概率和归一化熵。熵只描述分布集中程度，**不是正确率**；概率未经业务校准。`temperature` 是显式温度缩放参数，设置它本身不构成校准。小模型可能回答错误；本项目第一版验收的是计算、复用与接口，不将演示结果包装成通用决策质量保证。

## 缓存、批处理与边界

- context 与 question 分别分词；暖请求不重新序列化、分词或 prefill 整份 state。问题分词缓存最多 512 项。
- KV 在进程内跨请求保留，默认最多 32 个 context、1 GiB KV、600 秒滑动 TTL，按 LRU 淘汰。参数可通过 `serve --max-contexts/--cache-mib/--ttl` 调整。
- KV 不写磁盘，重启后 ID 不可用，需重新编译。模型权重缓存与运行时 KV 是不同层次。
- 缓存字节预算只约束保留的前缀 KV；模型、临时分支、激活和 MLX 分配器缓存另外占内存。默认 8 题一批，可调 `--batch-size`。
- MLX 按后缀 token 长度分组，不为 padding 或无关位置计算输出头。每个 batch 使用独立物理 KV 分支，绝不修改已保存的前缀。
- 一个模型操作占用一把锁。一个请求内可以 GPU batch，多 HTTP 请求目前依次执行；没有声称实现 continuous batching。
- 完整输入默认最多 8192 token，超限拒绝；不静默截断。空题集、重复选项、非有限数值、错误字段会返回 422；缺失、过期、淘汰和跨 namespace ID 返回 404。
- namespace 提供本地隔离，不是身份认证。此版只提供本机服务。
- 支持 dense、未量化的 Qwen3。模型适配显式检查；没有假装兼容所有 HF 模型。HF reference 默认 CPU FP32、串行执行。

`projection="selected"` 在词表投影前取出 A–P 对应权重；`projection="full"` 计算完整词表后取相同标签。两者用于对照。这是次要优化，核心仍是避免重复 prefill；共享词表权重继续保留以处理输入 token。

## 验证与 benchmark

```bash
# 普通回归，无需权重或 GPU。
uv run pytest -m 'not model'
uv run ruff check .

# 实际模型验证：MLX/HF、所有候选分数、缓存分支、16 题批处理。
uv sync --extra mlx --extra hf --extra dev
BUSBAR_RUN_MODEL_TESTS=1 uv run pytest -m model -s

# 比较 fresh/cached、serial/batch、full/selected；模型加载不计入延迟。
uv run busbar benchmark --context-tokens 4096 --questions 16 --repeats 3 \
  --output local-results/mac-4096-16.json
```

benchmark 保留硬件、依赖版本、精度、真实 prefix/suffix token 数、完整决策结果、耗时和结果差异；输出文件不会覆盖已有结果。`cold_compile_and_fanout` 包含一次 context prefill；暖路径不包含它。不同执行形状可能存在浮点差异，报告不掩盖选项变化。延迟数据来自合成 workload，不能推断模型质量或生产吞吐。

详见 [第一版设计与验收范围](docs/design.md) 和 [本机验收记录](docs/acceptance.md)。

## 后续阶段

Backend 协议把 context 生命周期与物理 KV 执行分开。Linux 阶段再接入 vLLM/SGLang 的 prefix cache、continuous batching 与 KV offload；Mac v0.1 不包含未验证的 vLLM 占位实现。训练专用决策头、量化、分布式 KV、跨进程恢复和校准数据集属于后续阶段。

本项目是独立的决策 runtime，不是 TypeSafe Jev 的官方实现，也不冒用其接口兼容性或性能结论。依赖 [MLX-LM](https://github.com/ml-explore/mlx-lm)、[Transformers](https://github.com/huggingface/transformers)，使用 [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) 模型。
