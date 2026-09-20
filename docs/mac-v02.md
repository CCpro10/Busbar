# Mac 3B、vLLM Metal 与评测记录

2026-09-21，北京时间。Apple M4 Pro，48 GiB 统一内存，macOS 15.6.1。
GoLand 已正常退出；测量时依次运行模型，未同时运行另一份推理服务。

## 模型与运行环境

- 新增 `Qwen/Qwen2.5-3B-Instruct`，3,085,938,688 个参数，未量化，固定 revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`。权重约 6.17 GB。
- 0.6B 对照固定 `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`。两者属于不同代模型，结果不是单独控制参数量的实验。
- 原生环境：Python 3.13.13、MLX 0.32.2、MLX-LM 0.31.3、Transformers 5.17.0。
- Metal 环境：Python 3.12.13、vLLM 0.29.0+cpu、vLLM Metal 0.29.0、MLX 0.32.1、
  MLX-LM revision `9e6acca691e64d6d8bb808c328fcdea459099cca`、PyTorch 2.13.0。
  完整依赖见 `requirements-metal.lock`，官方预编译 wheel，未修改依赖源码。
- 跨后端主比较统一 BF16。原生 FP16 另用于输出头诊断；FP32 用于 MLX/HF 正确性回归。
- 3B、4096-token 前缀、16 题的原生 FP16 诊断中，MLX 活跃内存峰值约 7.40 GiB；
  单份前缀 KV 为 144 MiB。这是 MLX 分配器指标，不是整台机器或所有进程 RSS。
  vLLM Metal 的默认 0.35 内存比例在本机分配约 6.72 GB KV 池，另外还有模型和运行时内存。

## 参考哪些评测

使用 [SemIf authored144](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/benchmarks/data/authored144.jsonl)
的全部 144 条、36 个源场景及其四种变体，保留原始 state、问题、选项顺序、标签和出处。
使用 Busbar 的提示词，不等同于复现 SemIf 的原始 scorer。标签是源项目经过模型复核的合成标签，
不是人工仲裁的业务测试集。数据与 MIT 许可一并保留在 `benchmarks/data/`。

参考 [Typed Decision Bench](https://blobfish.ai/benchmarks/typed-decision-bench) 同时报告覆盖率、
准确率、Brier、NLL 和校准误差的做法。它使用不同的 5,387 条数据与分数归一化规则，不能把这里的
结果填进其排名。没有调用付费 Jev API，没有训练、修改评测标签或在测试集上拟合温度。

## 全量质量结果

全部 144 条都有有效分布，覆盖率 100%。温度均为 1，表中的准确率按原始金标计算。

| 模型 / 后端，BF16 | 正确数 | 准确率 | 平均各任务平衡准确率 | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B / MLX | 78/144 | 54.17% | 48.76% | 0.7547 | 2.7284 | 0.3332 |
| Qwen2.5-3B / MLX | 99/144 | 68.75% | 68.48% | 0.5937 | 3.2512 | 0.2935 |
| Qwen2.5-3B / vLLM Metal | 98/144 | 68.06% | 67.90% | 0.5967 | 3.2404 | 0.3050 |

平均各任务平衡准确率已用 SemIf 原始 `evaluate.py` 独立复算，三份结果一致。Brier 使用多分类
平方误差的总和，范围 0–2；NLL 使用 1e-12 下限；ECE 使用十个等宽置信度区间。

这批样例里 3B 更准，但 NLL 比 0.6B 更差：它对部分错误答案更加确信。候选内概率不能直接当作
业务正确率。退货演示中的 Boolean/Score 也仍有不合理判断，未通过修改示例来隐藏。

## 同条件延迟比较

合成延迟 workload：2048-token 公共前缀，8 道题，每题后缀 51 token，batch size 8，
3 次正式测量取中位数。模型加载、形状预热不计入；结果保留所有逐轮时间和候选分布。
它不是 SemIf 的 37×21 workload，也不是质量测试。

| 执行方式 | 0.6B / MLX BF16 | 3B / MLX BF16 | 3B / vLLM Metal BF16 |
|---|---:|---:|---:|
| 每题重新计算整个输入，逐题 | 2.831 s | 13.832 s | 14.637 s |
| 每题重新计算整个输入，批量 | 2.786 s | 13.623 s | 14.486 s |
| 仅复用 state，逐题 | 0.218 s | 0.627 s | 0.698 s |
| 仅复用 state，批量 | 0.154 s | 0.432 s | 0.482 s |
| 仅复用 state，批量、完整词表 | 0.155 s | 0.437 s | 同上一行 |
| state 和问题都完全重复 | 未启用问题 KV 缓存 | 未启用问题 KV 缓存 | 0.142 s |
| 冷缓存：编译 state + 批量判断 | 0.596 s | 2.251 s | 2.235 s |

MLX 默认仅计算候选行；vLLM Metal 使用完整词表并读取指定 token 的 log probabilities。
相对自身 fresh 逐题路径，3B 暖缓存加速约为 MLX 31.98×、vLLM Metal 30.35×。
本机这组低并发测试没有证明 vLLM Metal 比原生 MLX 快；它提供的是已经接通的分页缓存与引擎执行路径。
HTTP 请求目前仍串行提交，不能把一个请求内的批量测试称为跨请求 QPS 测试。

为了保证公平，vLLM 的每次 state-only 测量前都清空 APC 并重新预热 state，预热时间不放进暖路径。
fresh 请求使用独立 cache salt，实际命中为零。引擎报告：

- fresh batch 实算 16,792 个输入 token；命中 0。
- state-only batch 命中 16,384 个前缀 token，实算 408 个后缀 token。
- 完全重复 batch 命中 16,768 个 token，只剩 24 个输入 token 重算；这是另一种工作负载。
- vLLM 每题仍请求一个生成 token，并在响应 `backend_details.generated_tokens` 中披露。

## 候选词表头能省多少

原生 3B FP16，4096-token 前缀、16 题，30 轮随机交替 selected/full，模型独占 GPU。

| 测量 | 只算候选行 | 完整词表 |
|---|---:|---:|
| 普通执行总耗时，中位数 | 786.61 ms | 792.29 ms |
| 普通执行 p95，最近秩 | 791.20 ms | 798.59 ms |
| 单独输出头，最后一个真实 batch，中位数 | 0.232 ms | 2.821 ms |

输出头本身约快 12.2×，端到端约省 0.7%。强制阶段同步的诊断中，16 题的 Transformer 约
764 ms，KV 分支构建约 40.5 ms，候选投影约 0.63 ms。阶段同步改变执行方式，不能把阶段时间
相加后当作原始请求延迟。单次矩阵计算替代逐行投影的头部微测量约 0.148 ms，收益只有不到
0.1 ms/batch，因此这次没有为了微小局部收益重构生产路径。原始候选投影与完整词表的分数在
这组 FP16 诊断中一致。

相同条件下，0.6B 的 30 轮复测为 selected 440.82 ms、full 443.59 ms，端到端约省 0.6%；
头部单独为 0.233 / 1.540 ms。本次未复现旧记录中 full 更快的现象，但不能据此断言旧差异的唯一原因。
0.6B 的阶段诊断中，KV 分支复制约 110.85 ms，Transformer 约 330.71 ms。其 4096-token
前缀 KV 为 448 MiB，反而大于这个 3B 的 144 MiB：两者 KV head 数分别为 8 / 2，层数为
28 / 36，head dimension 都为 128。KV 大小由这些结构参数决定，不能只按模型总参数量推断。
下一步性能工作应优先研究原生 KV 分支复制和执行形状，而不是把输出头的局部倍数当作整体加速。
这里的普通执行计时覆盖后端 `score()`，包含分支、Transformer、投影和读回，未计入 HTTP 传输。

## 精度与缓存差异

当前 vLLM Metal 普通权重加载器保留 checkpoint dtype，`--dtype float16` 不足以证明权重已转成
FP16。因此初始混合精度试跑不进入主表。后端现在默认且仅验收 BF16，通过具名 worker RPC
检查实际权重和 KV 类型；检查覆盖分页 attention wrapper 内部参数。没有启用不安全的 pickle RPC。
3B 主性能报告确认了全部 3,085,938,688 个参数和 KV 都是 BF16。
质量报告生成时检查尚未遍历 attention wrapper 的私有内层，参数计数较少；之后补齐只读检查，
HTTP 与性能报告均记录完整数量。这个变动未修改模型权重或推理路径。

统一 BF16 后，MLX 与 vLLM Metal 在质量集有 1/144 个 argmax 不同，最大候选概率差约 0.340。
再挑选差异最大的 6 条做诊断，fresh/cached 对照中，MLX 没有选项变化但最大概率差约 0.427，
vLLM Metal 有 1 条选项变化、最大概率差约 0.400。这 6 条是事后挑选的数值诊断，不能用于估计
总体错误率。BF16 改变执行形状时的概率漂移是真实限制，不能宣称三个后端的概率逐位等价。

独立 FP32 回归使用同一 3B checkpoint 验证三类决策、完整/候选投影、fresh/cached、16 分支、
问题重排及重复调用。MLX fresh 与 HF CPU fresh 最大 logit 差 0.002438，小于 0.005 门限；
MLX fresh/cached 最大 logit 差 0.00008393，小于 0.002 门限。它证明受测输入上的实现一致性，
不替代质量评估，也不是 BF16 vLLM 的严格数值等价证明。

## 接口与工程验收

26 项无需模型的回归通过；0.6B 和 3B 的真实模型 FP32 回归分别通过。Ruff 检查、格式检查、
锁文件检查、wheel/sdist 构建和 Metal 环境的 160 项依赖兼容性检查均通过。

最终 3B / vLLM Metal 常驻服务完成 17 次真实 HTTP 请求：三类决策、跨请求复用、64 题批量、
重试、版本更新、列表筛选、幂等删除、清空、空题集、跨 namespace、删除后访问及不支持的
selected 投影拒绝。健康接口核对完整 BF16 参数数量，OpenAPI 版本为 0.2.0。
额外输入边界验收验证 8191 token 返回 200、8192 token 返回 422；引擎 8192-token 预算中
需要预留一个输出 token。测试创建的上下文均已删除。

## 运行与复现

```bash
uv sync --extra mlx --extra hf --extra dev
bash scripts/setup-metal.sh

uv run busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 --output local-results/quality-new.json
.venv-metal/bin/busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --backend vllm-metal --model Qwen/Qwen2.5-3B-Instruct --output local-results/metal-quality-new.json

uv run busbar benchmark --model Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 \
  --context-tokens 2048 --questions 8 --repeats 3 --output local-results/latency-new.json
.venv-metal/bin/busbar benchmark --backend vllm-metal --model Qwen/Qwen2.5-3B-Instruct \
  --context-tokens 2048 --questions 8 --repeats 3 --output local-results/metal-latency-new.json

uv run busbar profile --model Qwen/Qwen2.5-3B-Instruct \
  --context-tokens 4096 --questions 16 --repeats 30 --output local-results/profile-new.json
BUSBAR_RUN_MODEL_TESTS=1 BUSBAR_TEST_MODEL=Qwen/Qwen2.5-3B-Instruct \
  uv run pytest -m model
```

原始数据位于 `benchmarks/results/mac-v02/`；`SHA256SUMS` 保留校验值，输出命令拒绝覆盖已有文件。
Qwen2.5-3B 权重许可见其模型卡，Busbar 的 MIT 许可不改变模型许可。

## 已知运行时限制

PyTorch 2.13.0 的 `torch.accelerator.empty_host_cache()` 在这台 macOS 15.6.1 机器上单独调用会
段错误；vLLM worker 退出时触发同一路径。上游 [vLLM Metal #765](https://github.com/vllm-project/vllm-metal/pull/765)
也记录了同版本、同 macOS 系列的问题。推理与常驻服务通过验收，但不宣称 worker 可以干净退出。
没有伪造成功关闭、修改依赖内部代码或把关闭阶段报错隐去。
