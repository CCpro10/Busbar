# 验证、评测与可复现对照

Busbar 分别回答三个问题：实现是否正确、给定任务是否答对、给定工作负载是否更快。三者使用不同证据。先安装 [开发环境](../CONTRIBUTING.md)，再选择相应验证层级；模型任务请串行运行，避免多个测试抢占同一块 GPU。

## 正确性回归

普通回归使用轻量后端检查接口、缓存生命周期、非法输入、数据划分和文件完整性；Mac 安装 MLX 后还会运行真实梯度与张量加载测试，不需要下载底座。

```bash
uv run pytest -m 'not model'
uv run ruff check .
uv run ruff format --check .
uv run ruff check src --select C901 --config 'lint.mccabe.max-complexity=10'

# 实际 0.6B：MLX/HF、selected/full、fresh/cached、所有候选和 16 题分支。
BUSBAR_RUN_MODEL_TESTS=1 uv run pytest tests/test_models.py -s

# 加载自己的头：候选重排、动态数量、独立缓存、禁止词表投影。
BUSBAR_TEST_HEAD=models/heads/support-v1 uv run pytest tests/test_head_models.py -s

# 另一个终端先运行 busbar serve，再用真实 HTTP 客户端检查生命周期。
uv run python examples/http_smoke.py --help
```

真实模型测试需要预装 `--extra mlx --extra hf`；训练头测试读取 `BUSBAR_TEST_HEAD` 指定的版本。未设置开关时这些测试会明确 skip，CI 通过不能替代真实模型验收。数值容差写在测试及保存的报告中；必须同时检查全部候选分数与最终选择。

## 任务质量

```bash
# 公开 SemIf 格式数据；保持模型、精度、数据和提示模板可追溯。
uv run busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --model Qwen/Qwen3.5-4B --dtype bfloat16 \
  --output local-results/quality-qwen35.json

# 本地业务：同一份独立测试集比较两个读出方式。
uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --dtype float32 --output local-results/quality-quick.json
uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --head models/heads/support-v1 --output local-results/quality-head.json
```

每个 JSONL 文件只读取一次，先验证所有行，再把捕获的原始字节同时用于解析与 SHA256。记录保留逐题标签、答案、概率、来源以及 accuracy、NLL、Brier、ECE；中途异常终止整次运行，不把部分答案当作完整覆盖。评测时间包含每个 state 的 prefill 和决策，不包含模型加载与独立预热。

训练头会拒绝与其训练/验证记录重叠的原生评测数据，包括继续训练的历史。`--allow-training-data` 只用于检查拟合，报告会明确标注不能作为独立测试。数据来源与标签约束见 [数据说明](../benchmarks/data/README.md)。自动去重只能发现记录 ID、group 和精确输入重叠，不能识别语义改写或错误的人工分组。

## 延迟与阶段计时

```bash
uv run busbar benchmark --context-tokens 4096 --questions 16 --repeats 3 \
  --output local-results/latency-4096-16.json
uv run busbar benchmark --head models/heads/support-v1 \
  --context-tokens 4096 --questions 16 --repeats 3 \
  --output local-results/latency-head-4096-16.json
uv run busbar profile --context-tokens 4096 --questions 16 --repeats 30 \
  --output local-results/projection-profile.json
```

benchmark 比较 fresh/cached、serial/batch；词表后端还对照 selected/full。报告保留每次测量的完整结果、实际 token 数、模型 SHA、精度、版本、硬件及结果偏移。`cold_compile_and_fanout` 包含一次 prefill；warm 路径不包含。合成退货 workload 只测延迟，不测业务质量，也不代表并发生产吞吐。

vLLM Metal 每次 warm 测量前重置 APC、只预热 state，重复整个问题的缓存收益独立列为 `cached_exact_repeat`。真实命中数来自引擎报告；Busbar snapshot 复用与物理 KV 命中不是同一件事。

profile 只支持原生 MLX 词表模式。普通端到端计时用于延迟结论；分阶段计时会插入 GPU 同步屏障，独立词表头微测量排除了 Transformer 工作，都不能冒充端到端收益。每次运行使用新的输出文件，原子发布且不覆盖已有报告。

## 对比其他项目

[对照脚本](../benchmarks/compare_projects.py) 和 [汇总脚本](../benchmarks/summarize_comparison.py) 保存 SemIf、NanoJev 与 Busbar 的实际输出。可比性取决于相同 checkpoint、dtype、数据、提示模板、候选语义及执行路径；换了任一项，就要在结论旁说明。

- [0.3 本机报告](mac-v03.md)：较新小模型、SemIf/NanoJev 对照、外部 WANLI，以及 Metal 失败记录。
- [0.4 本机报告](mac-v04.md)：三个底座训练头与快捷模式的质量/速度对照，NanoJev 原版与功能对齐两种口径。
- [0.4.1 质量整理](quality-v041.md)：本次代码改进与兼容性回归，不新增通用质量或速度排名。
- 历史记录：[0.2](mac-v02.md)、[0.1](acceptance.md)；每批公开结果保留 SHA256 清单。

例如，0.4 的小型客服测试上，0.6B、MiniCPM5、Qwen3.5 的头分别达到 68.75%、89.58%、100%，对应快捷模式为 85.42%、97.92%、100%。这说明选择头的训练收益要用数据证明；结构更原生不自动意味着更准或更快。样本仅 16 个工单、48 道派生问题，不能外推成通用榜单成绩。完整设置、耗时和项目对比见上述报告。
