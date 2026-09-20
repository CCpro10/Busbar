# Busbar v0.1：上下文复用优先

第一版的完成条件是：独立 Git 项目与公共 GitHub 仓库；这台 Apple Silicon Mac 上真实运行小模型；一次 context prefill 支持跨请求、多问题独立分支；给出可复现的正确性与性能证据。资料中的 vLLM、SGLang、TP、多层 KV offload、训练专用头等属于后续服务器与模型阶段，不在本版伪造实现或测量。

## 分层

`schemas` 定义 Boolean/Choice/Score 与上下文生命周期；`compiler` 管理稳定序列化、token 边界及问题分词缓存；`Runtime` 管理 snapshot 身份、namespace、TTL、LRU 和生命周期；`Backend` 管理物理 KV、前向计算、批处理与输出头。当前有两种真实实现：Mac MLX 与 CPU HF reference。

Compiler 固定 `busbar-qwen3-v1`。稳定 system + context user message 在 `<|im_end|>\n` 后结束；动态 question 从 `<|im_start|>user\n` 开始。两段以 token IDs 拼接，题目不会进入已保存的前缀。候选标签必须是单 token，并验证追加标签不会重新切分答案边界。

snapshot ID 由 namespace、规范 JSON state、instructions、模型与 tokenizer revision、后端精度及 compiler 版本共同 SHA-256 计算。改变任一项都会得到新的身份；请求方修改原始字典不会影响已编译的 tokens/KV。

## 生命周期与一致性

| 场景 | 约束与验证 |
|---|---|
| 重复创建 | 按内容复用，仅一次分词/prefill；并发调用也只创建一次 |
| 编辑 | 创建新的不可变版本，成功后返回新 ID；旧版本受正常 TTL/LRU 管理 |
| 回退 | 使用仍存活的旧 ID；淘汰后可重新提交旧 state 重建 |
| 删除/清空 | 释放 runtime 的引用，之后请求明确 404；幂等删除不报错 |
| 列表/筛选 | 只返回指定 namespace 的实时元数据，支持 ID 前缀过滤 |
| 空状态 | JSON null、空对象是合法无额外上下文；空问题集不是合法判断请求 |
| 请求失败 | 全部题目先验证，再执行；不返回部分答案，已有前缀不被分支修改 |
| 缓存超限 | 单上下文超预算则拒绝；正常入池根据数量与字节双限制淘汰 LRU |
| 过期/重启 | 单调时钟控制 TTL；进程退出不保留 KV，重启后明确需重新编译 |
| 输入超长 | context 或 context+question 超限时拒绝，禁止截断后假装同一输入 |

当前锁覆盖创建、推理和删除，避免 GPU 模型并发执行与生命周期竞态。锁也是明确的吞吐上限；下一阶段应在真实服务器后端实现调度，不在本地适配层复制 Paged KV allocator。

MLX fanout 按相同 suffix 长度合批，通过原生 `BatchKVCache.merge` 分配新的分支缓存。这样每题都有完整独立的可写 KV，已保存的 prefix 不进入 mutating forward。该策略会复制前缀 KV 到 batch 缓冲区；它复用计算，但不等同于 vLLM 的物理分页共享。前缀很短时，分支复制、启动和批处理成本可能大于所省 prefill。

## 数学与质量边界

先计算候选单 token 的原始 logits，再仅在候选集合内 softmax。Score 是显式数值等级的期望。FP32、FP16、batch shape、attention kernel 与 prefill 分块可能改变浮点累加；测试必须比较每个候选及 argmax，不能仅以概率和为 1 判定正确。

HF reference 可以 full vocabulary，也可以 selected rows。MLX 与 HF 使用同一官方 checkpoint 及 tokenizer revision，模型特定适配仅接受 dense Qwen3。后续支持量化模型时，需要保留原量化语义并重新建立数值误差门限。

## 验收门

1. Python/HTTP 生命周期回归覆盖上述边界，确保验证错误不执行部分推理。
2. 实际模型跑通 Boolean、Choice、Score；完整记录模型回答，包括回答错误。
3. 实际 MLX selected/full、fresh/cached、问题顺序变化与 16 题 batch 均检查候选分数和选择。
4. 同 checkpoint 的 HF CPU FP32 与 MLX FP32 做交叉参考；公开容差和差异。
5. Mac FP16 约 4096-token prefix、16-question workload 比较六种路径，保留原始记录。64 题验证分块与上限行为。
6. 启动真实 HTTP 进程，用客户端完成创建、跨请求复用、判断、版本化、删除及失效检查。
7. 代码格式/静态检查、普通测试、实际模型测试、可安装包构建通过；提交并推送到公共仓库，确认远端 SHA 与本地一致、工作区干净。
