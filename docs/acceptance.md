# 本机验收记录

核验日期：2026-09-20。本页数据来自本机真实运行；没有调用云端模型，没有下载或使用大型模型。

## 环境与模型

| 项目 | 实际值 |
|---|---|
| 机器 | Apple M4 Pro，48 GiB 统一内存 |
| 系统 | macOS 15.6.1，arm64 |
| Python | 3.13.13 |
| 模型 | Qwen/Qwen3-0.6B |
| 模型和 tokenizer revision | c1899de289a04d12100db370d81485cdf75e47ca |
| MLX / MLX-LM | 0.32.2 / 0.31.3 |
| Torch / Transformers | 2.10.0 / 5.17.0 |
| 默认 Mac 精度 | FP16；交叉正确性测试使用 FP32 |

依赖由 `uv.lock` 固定。仓库仅保存源码、合成示例和结果，不保存模型权重。

## 功能与数值验证

普通回归：`uv run pytest -m 'not model' -q`，**17 passed**。覆盖 JSON 规范化身份、namespace 隔离、重复创建、不可变版本及回退、删除/清空、TTL、LRU、字节预算、过大 context 拒绝、失败请求不执行部分题目、并发创建一次 prefill、suffix token cache、调用者字典变更不污染快照、非法标签与数值、Score 数学语义、HTTP 生命周期与异常。

真实模型：`BUSBAR_RUN_MODEL_TESTS=1 uv run pytest -m model -s -q`，**1 passed**。这个集成测试包含多项独立断言：MLX full/selected、fresh/cached、问题顺序反转、16 题分支进入两个真实 batch、重复调用后前缀不变、HF CPU 参考，以及每个候选的 logits/概率/argmax。

| FP32 比较 | 实测最大 logits 绝对差 | 实测最大概率绝对差 | 最大分数选项 |
|---|---:|---:|---|
| MLX fresh/full vs MLX cached/selected | 0.00005722 | 0.000005015 | 3/3 一致 |
| MLX fresh/full vs HF fresh/full | 0.00008965 | 0.000007164 | 3/3 一致 |

事先设置的绝对容差为：MLX 输出投影 0.001、缓存执行 0.002、MLX/HF 交叉比较 0.005。原始候选分数、概率及配置保存在 [FP32 正确性记录](../benchmarks/results/mac-fp32-correctness.json)。这些门限属于本次指定模型、精度和样例的验收，不是对其他模型或全部输入的统一保证。

## 真实 HTTP 验收

启动 `uv run busbar serve --port 8787` 后，通过独立客户端执行：

```bash
uv run python examples/http_smoke.py --output local-results/http-smoke.json
```

**16 次真实 TCP/HTTP 请求通过**，覆盖驻留模型健康检查、创建/重复创建、跨请求复用、三种判断、64 题 fanout、再次判断、空题集 422、跨 namespace 404、新版本创建、前缀过滤、幂等删除、删除后 404 及清空。

64 题被拆成 8 个实际 GPU batch；仅提交 3,968 个 suffix token，复用前缀位置合计 7,552 个。该检查使用 118-token 的示例前缀，不能冒充 4096-token 的 64 题性能测量。[HTTP 原始记录](../benchmarks/results/mac-http-smoke.json)

CLI 同样完成实际模型运行：`uv run busbar run examples/returns.json`。[FP16 CLI 原始输出](../benchmarks/results/mac-fp16-example.json)

## 4096-token、16 题实测

```bash
uv run busbar benchmark --context-tokens 4096 --questions 16 --repeats 3 \
  --output local-results/mac-4096-16.json
```

真实 prefix 为 **4096 token**，每题 suffix 为 **55 token**，保留前缀 KV 为 **469,762,048 bytes（448 MiB）**。模型驻留、已预热，3 次测量取中位数；输入是合成 workload，测量期间没有同时运行本任务的其他 GPU 实验。不是受控独占服务器，也不是质量评测。

| 路径 | 中位耗时 | 最小—最大 | 每次实际提交的 token 位置 |
|---|---:|---:|---:|
| 每题重算，逐题调用 | 13,722.97 ms | 13,645.47—13,820.38 ms | 66,416 |
| 每题重算，批量调用 | 13,631.13 ms | 13,561.19—13,635.36 ms | 66,416 |
| 复用前缀，逐题调用 | 730.51 ms | 636.47—743.15 ms | 880 |
| 复用前缀，批量调用，selected head | 858.17 ms | 493.91—921.45 ms | 880 |
| 复用前缀，批量调用，full head | 506.09 ms | 503.69—507.70 ms | 880 |
| 冷 context 编译 + 批量判断 | 1,609.20 ms | 1,568.49—1,644.14 ms | 4,976 |

暖批量 selected 相对 fresh 逐题的本次中位加速为 **15.99 倍**；包含一次 prefix 编译的冷批次相对 fresh 逐题约 **8.53 倍**。这里加速来自避免重复 prefill，不能推广为任意任务、模型或硬件的固定倍数。暖 selected 有显著波动，而且本次 full head 更快；本结果不支持“selected head 已获得稳定端到端提速”的结论。

每条路径有 16×3=48 次选项比较，相对首次 fresh 均未改变最大分数选项；FP16 缓存路径最大概率差约 **0.01220**。不能将“argmax 一致”表述成“概率完全相同”。计数在不同批次中分别为：`16×(4096+55)=66416`、`16×55=880`、`4096+16×55=4976`。[完整 benchmark 记录](../benchmarks/results/mac-fp16-4096-16.json)

## 已观察到的质量限制

售后示例中，0.6B 模型将路由选为 `returns`，Score 约为 1.998；但 Boolean “此订单是否符合退货条件”返回 `false`，与示例条件不符。MLX FP32 和 HF FP32 参考也得到相同错误选择，因此不能把该错误归因于缓存或四行投影。原始输出完整保留。

这说明当前可验证的成果是上下文复用 runtime，而不是已通过业务准确率验证的退货系统。更换模型、任务提示、标签设计与校准需要单独的质量数据集和对照实验。

## 发布检查

`ruff check`、`ruff format --check`、普通回归、真实模型集成、HTTP/TCP、CLI、`uv build` 均在本机通过。Linux CI 只运行不依赖模型的契约回归及包构建；不能把它标记为 Linux GPU 或 vLLM 验收。Git 提交、公共仓库和 CI 的最终状态以仓库实际记录为准。
