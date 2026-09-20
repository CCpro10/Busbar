# 接口与运行时约束

Busbar 的公共入口是 `ContextSpec`、`DecisionRequest` 和 `Runtime`。CLI、Python、HTTP 共用同一套验证与决策逻辑。完整混合题目示例见 [returns.json](../examples/returns.json)；服务启动后，`/docs` 和 `/openapi.json` 给出当前安装版本的请求与响应 schema。

## HTTP 接入

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

所有结果包含候选分数（`logit_space` 区分词表原始 logits、词表 log probabilities 与训练头分数）、候选内 softmax 概率和归一化熵。熵只描述分布集中程度，**不是正确率**；概率未经业务校准。`temperature` 是显式温度缩放参数，设置它本身不构成校准。小模型可能回答错误；演示结果不构成通用决策质量保证。

## 缓存、批处理与边界

- context 与 question 分别分词；暖请求不重新序列化、分词或 prefill 整份 state。问题分词缓存最多 512 项。
- 默认最多 32 个 context、600 秒滑动 TTL，按 LRU 淘汰；MLX/HF 另外限制 1 GiB 保留 KV。参数可通过 `serve --max-contexts/--cache-mib/--ttl` 调整。
- vLLM Metal 的物理 KV 由引擎管理，`cache_bytes=0` 表示 Busbar 不直接持有 KV，不表示引擎不占内存。引擎预算使用 `--memory-fraction`。删除、过期或淘汰会立即撤销 snapshot 访问；对应物理块之后按引擎策略淘汰。每个新 snapshot 使用独立 cache salt，防止跨 namespace 复用；全局 `Runtime.clear()` 也重置引擎 APC。
- 创建响应中的 `reused=true` 表示 Busbar 复用了已有 snapshot 和 token，不能证明引擎物理块仍驻留；vLLM 决策响应会按实际命中记录 token 数，块被淘汰后由引擎重新计算。
- KV 不写磁盘，重启后 ID 不可用，需重新编译。模型权重缓存与运行时 KV 是不同层次。
- 缓存字节预算只约束保留的前缀 KV；模型、临时分支、激活和 MLX 分配器缓存另外占内存。默认每批 8 条路径：快捷模式一题一条，选择头每个候选一条，可调 `--batch-size`。
- MLX 按后缀 token 长度分组，不为 padding 或无关位置计算输出头。每个 batch 使用独立物理 KV 分支，绝不修改已保存的前缀。
- 一个模型操作占用一把锁。一个请求内可以 GPU batch，多 HTTP 请求目前依次执行；没有声称实现 continuous batching。
- 原生后端完整输入默认最多 8192 token；vLLM Metal 从此预算预留一个输出 token，因此最多接受 8191 个输入 token。超限拒绝，不静默截断。空题集、重复选项、非有限数值、错误字段会返回 422；缺失、过期、淘汰和跨 namespace ID 返回 404。
- namespace 提供本地隔离，不是身份认证。此版只提供本机服务。
- MLX/HF 支持 dense、未量化的 Qwen2/3/3.5 和 Llama 架构（含 MiniCPM5）；vLLM Metal 当前验证 Qwen2/3 和 MiniCPM5，尚未开放 Qwen3.5。模型适配显式检查；HF reference 默认 CPU FP32、串行执行。
- Qwen3.5 的缓存同时包含注意力 K/V 与递归状态，分支必须独立复制两者。MLX-LM 固定到包含 GDN 归一化修复的提交；其 `A_log` 保持 FP32。FP32 参考加载会在权重转换运算前提升精度，避免继承 BF16 舍入误差。

MLX/HF 的 `projection="selected"` 在词表投影前取出 A–P 对应权重；`projection="full"` 计算完整词表后取相同标签。词表模式下，默认 `auto` 在 MLX/HF 使用 `selected`，在 vLLM Metal 使用 `full`；加载训练头时使用 `head`。候选行投影是次要优化，核心仍是避免重复 prefill；共享词表权重继续保留以处理输入 token。

## 失败时如何处理

| HTTP 状态 | 含义 | 应用应采取的动作 |
|---|---|---|
| 404 | ID 缺失、过期、被淘汰，或 namespace 不匹配 | 使用原始上下文重新创建 snapshot |
| 413 | 单份上下文 KV 超过保留预算 | 减少上下文，或增加 `--cache-mib` |
| 422 | 请求格式、候选描述、输入长度或投影模式无效 | 根据错误信息修正请求；问题和候选描述不能只有空白 |
| 502 | 后端返回非法分数、计数或响应元数据 | 检查服务端日志和后端实现；已有 snapshot 仍可重试 |

没有部分成功响应：运行时先检查整批问题的结构、标签与 token 长度，再执行推理。后端输出也必须通过形状、有限数值和计数检查，之后才组装结果。普通 JSON 验证失败不会触发模型前向；与 tokenizer 有关的检查发生在模型已驻留之后。

上下文 `state=null` 或空对象是合法空状态；空题集被拒绝。PUT 创建新版本，不修改旧 KV。回退可继续使用仍存活的旧 ID；若已经淘汰，就重新提交旧内容。列表只显示当前 namespace 中的存活版本，支持 ID 前缀筛选。删除是幂等的，进程重启后需重建所有 snapshot。

服务只绑定本机地址，没有身份认证或多租户授权。namespace 是缓存隔离范围，不能用它证明用户身份。部署边界与后端限制见 [后端指南](backends.md)，内部职责见 [架构说明](design.md)。
