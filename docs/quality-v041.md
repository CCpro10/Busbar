# 0.4.1：代码质量与兼容性整理

本次以 `8b3853e` 为基线，审查 CLI、HTTP、Runtime、数据集、选择头文件、训练及评测流程。主要目标是修复边界错误、拆开混合职责、补足项目说明。底座、候选编码版本、训练算法和 artifact schema 保持原有语义。

## 修复的问题与用户影响

| 原问题 | 用户可能遇到的结果 | 修复及回归 |
|---|---|---|
| 训练/评测先哈希、再打开文件解析，随后又读取来源 | 数据文件在执行中被编辑，报告可能记录了另一份内容的哈希 | 单次捕获字节，同时用于解析与哈希；模拟编辑验证评测一致性；训练源变更时拒绝发布 |
| 头文件校验后再次按路径读取 | 文件被替换后，实际加载的权重或训练来源未必是校验过的版本 | 从已验证字节解码；真实 MLX 测试在校验后替换文件，已捕获的版本仍正确，新加载明确失败 |
| 非有限模型分数通过 ValueError 冒泡 | HTTP 返回 422，把后端异常归为调用方错误 | 验证分数形状/有限性、计数和元数据；统一 BackendOutputError → 502，原 snapshot 可在恢复后重试 |
| CLI 在完整验证前加载模型，SemIf 缺失字段可能在前向后报错 | 错误文件触发下载/prefill，最后才失败，行定位不清晰 | run 整体契约和全体评测记录提前验证；报告文件名与行号；保留可读的外部 family/group 元数据 |
| JSON 直接写入目标文件 | 读者可能看到半份报告或 manifest | 同目录临时文件完整写入后原子、不覆盖地发布；测试并发唯一成功以及失败清理 |
| 空白候选描述被接收，缺失 EOS 抛出属性异常 | 无意义输入进入模型，编码器错误不清晰 | 统一非空白文本约束，明确验证 EOS 类型与值 |
| 评测推理失败时未明确释放当前 snapshot | 当前评测上下文可能保留到对象回收 | finally 删除当前上下文，注入后端错误验证 KV 引用与计数归零 |
| OpenAPI 版本常量仍为 0.3.0 | 安装版本与接口版本不一致 | 从安装包元数据读取，普通及真实 HTTP 验证均为 0.4.1 |

新增回归描述的是文件被编辑、后端恢复、并发发布、错误数据和旧版本加载等场景。哈希是文件完整性机制，不是签名或语义数据泄漏检测。改写文本与不正确的 group 划分仍需数据作者处理。

## 结构整理

`storage.py` 统一文件快照与原子 JSON 写入；`datasets.py` 负责两种 JSONL 的验证、标签和来源；`head_artifact.py` 负责版本格式与完整性。`training.py` 保留流程、预检、继承记录和发布策略，张量特征提取/优化移入 `mlx_training.py`。

CLI 的加载、run、serve、evaluate、测量、训练分别具有入口函数；HTTP 的错误映射、状态、上下文和决策路由分组；Runtime 的问题预检、后端执行与结果归并分离。benchmark 的策略、测量、汇总、环境记录与 profile 的独立输出头测量各自独立。

这些拆分减少混杂职责，未增加框架或依赖。作为可读性的辅助指标，CLI `main` 从 92 行降到 25 行，`create_app` 从 71 行降到 11 行，`benchmark` 从 166 行降到 56 行。CI 增加生产源码函数复杂度不超过 10 的检查；函数长度不是判断质量的唯一依据。

## 已执行的验证

原始结果见 [本次证据目录](../benchmarks/results/mac-v041/)，汇总见 [summary.json](../benchmarks/results/mac-v041/summary.json)。

| 层级 | 结果与范围 |
|---|---|
| 普通回归 | Mac 上 72 passed；2 个真实模型用例另行开启。包含真实小型 MLX 梯度与张量加载测试 |
| MLX/HF | Qwen3-0.6B：全部候选、selected/full、fresh/cached、16 题批量通过 |
| 旧头兼容 | 0.6B、MiniCPM5-2B、Qwen3.5-4B：重载、独立缓存、候选重排、扩展至 16 个候选、禁止执行词表头均通过 |
| 从头训练 | 相同数据、seed 42、40 轮，最佳 epoch 6；每轮训练/验证记录与 0.4 完全相同，最终权重 SHA256 完全相同 |
| 继续训练 | 旧 0.6B 头再训练 2 轮；验证未改善，保留旧参数，best_epoch=0；继承链包含旧、新四份划分记录 |
| 业务质量回归 | 同一份 48 题测试，快捷模式 85.42%、选择头 68.75%；两条路线全部候选分数与最终选择都与 0.4 完全相同 |
| SemIf 格式 | 144 行完整校验与执行，coverage=1；不把本次结果当成新增跨项目排名 |
| benchmark/profile | 128-token、2 题小规模实跑；覆盖快捷、选择头策略和阶段计时报告，属于命令回归 |
| 真实 HTTP | 17 个请求与 64 题批量通过；OpenAPI 报告 0.4.1；测试服务已正常退出 |
| 静态与构建 | Ruff 检查/格式、复杂度、diff 空白检查及 wheel/sdist 构建通过 |

普通测试中的两条 warning 来自 Starlette/httpx 和 anyio 的上游弃用提示。本轮没有为消除警告升级或混装依赖。

## 复现主要步骤

```bash
uv sync --locked --extra mlx --extra hf --extra dev
uv run pytest -m 'not model'
BUSBAR_RUN_MODEL_TESTS=1 uv run pytest tests/test_models.py
BUSBAR_TEST_HEAD=path/to/existing-head uv run pytest tests/test_head_models.py

uv run busbar train-head examples/trainable_support/train.jsonl \
  --validation examples/trainable_support/validation.jsonl \
  --output models/heads/review-v041 --epochs 40
uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --head models/heads/review-v041 --output local-results/review-v041-test.json

# 一个终端运行服务，另一个运行客户端；完成后停止该服务。
uv run busbar serve --head models/heads/review-v041 --port 8787
uv run python examples/http_smoke.py --output local-results/review-v041-http.json
```

对照训练结果时比较 `head.safetensors` 的 SHA256 和 `training.json` 中的 history；时间字段会变化，因此整个 manifest 的 identity 可以不同。比较预测时按行 ID 对齐，并逐个候选比较 logits，不只看最终准确率。

本轮未重新运行 vLLM Metal 真实引擎，也未重跑全量跨项目性能实验。其适配契约测试通过，已有 [Metal 数值/退出限制](backends.md) 没有在本次修改中解决。此前原始结果保持不变，当前主要证据是原生 MLX、HF 参考与 HTTP 全流程。
