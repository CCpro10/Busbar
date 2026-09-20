# 0.4 可训练选择头：原始证据

说明与结论见 [本机报告](../../../docs/mac-v04.md)，训练/加载命令见 [选择头指南](../../../docs/trainable-heads.md)。结果包含错误预测，未按正确率挑选样本。

- `*-training.json`：冻结底座、实际提取特征并训练头的完整逐轮记录；包括初始指标、验证集选择、参数变化与数据划分。`*-manifest.json` 固定底座 revision、精度、编码器和权重哈希。权重本体不提交到仓库，这些元数据文件不能直接当作可加载目录。
- `*-quick-test.json` / `*-head-test.json`：相同 48 题的逐条预测、完整概率、误差指标及时间。48 题来自 16 个相关工单，不是独立大样本；单次质量评测的时间不代替速度 benchmark。
- `*-head-gate-final.json`：三个真实底座的 cached/fresh、重排/改 ID、16 候选扩展、16 题扇出及词表头调用禁止检查。`vocabulary-regression.json` 保留原有 0.6B 的 MLX/HF、完整/候选词表头回归。
- `head-http-smoke.json`：真实 TCP 服务的 17 次请求、64 题扇出，以及重复决策数值检查；最终状态 passed。随后正常停止测试服务以释放内存。
- `warm-start-*.json`：从真实旧头继续训练后，因验证没有改善而保留旧权重；保留了历史数据划分。`cli-negative-checks.json` 记录覆盖版本、底座错配和历史训练数据泄漏的拒绝。
- `minicpm5-*-speed.json`：512 token 前缀、16 道二选一题、FP32、3 次重复的全部输出与延迟。两个模式分别采用各自的编码器；测试合成问题仅用于速度观察。
- `semif-qwen35-test.json` / `nanojev-test.json`：固定上游源码的外部项目对照，记录其原始模型版本、精度和逐题结果。SemIf 使用 BF16 Qwen3.5；NanoJev 使用原游戏训练 checkpoint 的 MPS BF16 autocast。不同训练与运行配置不可直接解释为架构排名。
- `summary.json` 汇总原始质量结果，不替代逐条证据。`INPUTS.sha256` 固定本次样例和生成器，`SHA256SUMS` 固定此目录文件。

在仓库根目录校验：

```bash
shasum -a 256 -c benchmarks/results/mac-v04/INPUTS.sha256
cd benchmarks/results/mac-v04
shasum -a 256 -c SHA256SUMS
```

数据源在 `examples/trainable_support/`，其中 `test-categorical.jsonl` 由同一生成器把 Boolean/Score 转成具有同样候选含义的分类输入，供外部项目使用。原生 Busbar 对照直接读取 `test.jsonl` 的三类 API 契约。外部比较复用仓库的 `benchmarks/compare_projects.py`；SemIf 与 NanoJev 的安装和 Mac 移植边界沿用 [0.3 报告](../../../docs/mac-v03.md)，不改上游代码。
