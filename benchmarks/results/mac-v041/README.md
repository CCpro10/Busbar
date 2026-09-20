# 0.4.1 代码整理回归

M4 Pro、48 GiB，本地原生 MLX。基线提交 `8b3853e`，包版本 0.4.1；目的为验证重构和 bug 修复没有改变既有计算结果，不产生新的性能排名。

- `summary.json`：相同数据的逐题比较、训练权重/历史一致性、HTTP 结果及覆盖边界。
- `correctness-06b.json`：MLX/HF、selected/full、fresh/cached 和批量数值门禁。
- `*-old-head.json`：0.4 的三个底座选择头重载，包含词表投影禁用检查、重排、扩展候选和缓存独立性。
- `retrained-*` / `warm-start-*`：0.6B 重新训练与继续训练的完整记录和 manifest。模型权重留在本机，不随仓库发布。
- `quality-quick.json` / `quality-head.json`：相同 48 道测试题；所有候选分数及选择与 0.4 完全相同。
- `quality-semif.json`：144 行 SemIf 格式输入验证及完整推理。
- `benchmark-*-smoke.json` / `profile-smoke.json`：128-token 前缀、2 题，确认命令/策略/输出格式；不用于速度结论。
- `head-http-smoke.json`：17 个真实 HTTP 请求、64 题 fanout、重试数值检查。

校验：在本目录运行 `shasum -a 256 -c SHA256SUMS`。完整命令与问题说明见 [质量整理记录](../../../docs/quality-v041.md)。原有 0.3/0.4 证据不作修改。vLLM Metal 本轮仅运行普通适配契约回归，未重新启动真实引擎；之前记录的缓存数值和退出限制仍然成立。
