# Busbar

A context-reuse decision runtime for LLM agents.

Busbar 把上下文编译成可复用的 `ContextSnapshot`：后续 Boolean、Choice、Score 判断复用共同前缀。面向 Apple Silicon Mac，支持 Qwen3-0.6B、Qwen2.5-3B-Instruct；MLX 使用独立 KV 分支，vLLM Metal 使用引擎的分页前缀缓存，HF/PyTorch 提供 CPU 正确性参考。

M4 Pro、48 GiB 本机已跑通未量化 3B 和 vLLM Metal。相同 BF16、SemIf 144 条样例中，0.6B / MLX 准确率 54.17%，3B / MLX 为 68.75%，3B / vLLM Metal 为 68.06%。3B / vLLM Metal 在 2048-token 公共前缀、8 题测试中，从逐题重算的 14.64 秒降至复用前缀批量判断的 0.48 秒。完整条件、概率差异、退出限制和原始数据见 [0.2 本机报告](docs/mac-v02.md)；旧版数据见 [0.1 验收记录](docs/acceptance.md)。

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

运行较大的模型（约 6.2 GB 权重，本机 48 GiB 内存已验证）：

```bash
uv run busbar run examples/returns.json --model Qwen/Qwen2.5-3B-Instruct
```

这个型号自动使用固定 revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。其他型号必须显式指定 `--revision`；不会把 0.6B 的版本号误用于新模型。

## vLLM Metal

使用独立 Python 3.12 环境，避免 vLLM Metal 原生扩展需要的 MLX 版本与原生后端依赖冲突：

```bash
bash scripts/setup-metal.sh
.venv-metal/bin/busbar run examples/returns.json \
  --backend vllm-metal --model Qwen/Qwen2.5-3B-Instruct
.venv-metal/bin/busbar serve --backend vllm-metal \
  --model Qwen/Qwen2.5-3B-Instruct --port 8787
```

安装脚本使用官方 vLLM 0.29.0 / vLLM Metal 0.29.0 wheels，完整依赖版本记录在 `requirements-metal.lock`；安装后执行依赖兼容性检查。默认 BF16，`--memory-fraction 0.35` 限制引擎缓存预算。后端通过只读 worker 扩展验证真实权重类型，避免当前插件保留 BF16 权重、但命令行声称 FP16 的情况；MLX 对照跑分也要指定 `--dtype bfloat16`。先停掉同端口的旧服务，再启动新后端。

此后端真实执行完整词表头，每题请求一个生成 token，同时读取所有候选 token 的原始 log probabilities，随后在候选内重新归一化。它们与原始 logits 相差一个共同常数，softmax 比例保持一致；响应以 `logit_space="vocabulary_logprobs"` 标注。`projection="auto"` 自动选择 `full`；显式要求 `selected` 会拒绝，不声称已实现候选行投影。

**已知上游退出问题：** 当前 macOS 15.6.1 + PyTorch 2.13.0 环境在引擎退出清理时可能发生 `empty_host_cache()` 段错误；独立调用也能复现，上游 [#765](https://github.com/vllm-project/vllm-metal/pull/765) 记录了同类问题。推理与常驻服务可工作，暂不宣称干净的 worker 退出。没有修改依赖源码或屏蔽错误。

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

所有结果包含候选分数（`logit_space` 区分原始 logits 与词表 log probabilities）、候选内 softmax 概率和归一化熵。熵只描述分布集中程度，**不是正确率**；概率未经业务校准。`temperature` 是显式温度缩放参数，设置它本身不构成校准。小模型可能回答错误；演示结果不构成通用决策质量保证。

## 缓存、批处理与边界

- context 与 question 分别分词；暖请求不重新序列化、分词或 prefill 整份 state。问题分词缓存最多 512 项。
- 默认最多 32 个 context、600 秒滑动 TTL，按 LRU 淘汰；MLX/HF 另外限制 1 GiB 保留 KV。参数可通过 `serve --max-contexts/--cache-mib/--ttl` 调整。
- vLLM Metal 的物理 KV 由引擎管理，`cache_bytes=0` 表示 Busbar 不直接持有 KV，不表示引擎不占内存。引擎预算使用 `--memory-fraction`。删除、过期或淘汰会立即撤销 snapshot 访问；对应物理块之后按引擎策略淘汰。每个新 snapshot 使用独立 cache salt，防止跨 namespace 复用；全局 `Runtime.clear()` 也重置引擎 APC。
- 创建响应中的 `reused=true` 表示 Busbar 复用了已有 snapshot 和 token，不能证明引擎物理块仍驻留；vLLM 决策响应会按实际命中记录 token 数，块被淘汰后由引擎重新计算。
- KV 不写磁盘，重启后 ID 不可用，需重新编译。模型权重缓存与运行时 KV 是不同层次。
- 缓存字节预算只约束保留的前缀 KV；模型、临时分支、激活和 MLX 分配器缓存另外占内存。默认 8 题一批，可调 `--batch-size`。
- MLX 按后缀 token 长度分组，不为 padding 或无关位置计算输出头。每个 batch 使用独立物理 KV 分支，绝不修改已保存的前缀。
- 一个模型操作占用一把锁。一个请求内可以 GPU batch，多 HTTP 请求目前依次执行；没有声称实现 continuous batching。
- 原生后端完整输入默认最多 8192 token；vLLM Metal 从此预算预留一个输出 token，因此最多接受 8191 个输入 token。超限拒绝，不静默截断。空题集、重复选项、非有限数值、错误字段会返回 422；缺失、过期、淘汰和跨 namespace ID 返回 404。
- namespace 提供本地隔离，不是身份认证。此版只提供本机服务。
- 支持 dense、未量化的 Qwen2/3。模型适配显式检查；HF reference 默认 CPU FP32、串行执行。

MLX/HF 的 `projection="selected"` 在词表投影前取出 A–P 对应权重；`projection="full"` 计算完整词表后取相同标签。默认 `auto` 在 MLX/HF 使用 `selected`，在 vLLM Metal 使用 `full`。候选行投影是次要优化，核心仍是避免重复 prefill；共享词表权重继续保留以处理输入 token。

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

# 原始 144 条公开样例：质量、Brier、NLL、ECE 与逐条答案。
uv run busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 --output local-results/quality-3b.json
.venv-metal/bin/busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --backend vllm-metal --model Qwen/Qwen2.5-3B-Instruct \
  --output local-results/quality-3b-metal.json

# 对照正常执行与增加同步点的阶段计时，独占 GPU 运行。
uv run busbar profile --model Qwen/Qwen2.5-3B-Instruct \
  --context-tokens 4096 --questions 16 --repeats 30 \
  --output local-results/profile-3b.json
```

benchmark 保留硬件、依赖版本、精度、真实 prefix/suffix token 数、完整决策结果、耗时和结果差异；输出文件不会覆盖已有结果。`cold_compile_and_fanout` 包含一次 context prefill；暖路径不包含它。vLLM Metal 暖路径在每次测量前清空 APC、只预热 state，完整重复问题缓存单独列为 `cached_exact_repeat`。`backend_details.engine_cached_tokens` 来自引擎真实报告；`reused_prefix_tokens` 只统计其中的 context 部分。不同执行形状可能存在浮点差异，报告不掩盖选项变化。延迟数据来自合成 workload，不能推断模型质量或生产吞吐。

详见 [第一版设计与验收范围](docs/design.md) 和 [本机验收记录](docs/acceptance.md)。

## 后续阶段

Backend 协议把 context 生命周期与物理 KV 执行分开。Mac 已接入 vLLM Metal；Linux vLLM/SGLang、跨 HTTP 请求的 continuous batching、KV offload、训练专用决策头、量化、分布式 KV、跨进程恢复和概率校准属于后续阶段。

本项目是独立的决策 runtime，不是 TypeSafe Jev 的官方实现，也不冒用其接口兼容性或性能结论。依赖 [MLX-LM](https://github.com/ml-explore/mlx-lm)、[Transformers](https://github.com/huggingface/transformers)，使用 [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) 模型。
