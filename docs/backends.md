# 后端、模型与运行环境

## 选择后端

| 后端 | 环境 | 快捷词表模式 | 可训练选择头 | 物理前缀缓存 |
|---|---|---|---|---|
| 原生 MLX | Apple Silicon、macOS 15+ | selected / full | 训练与推理均支持 | Busbar 持有，分支时复制 KV/递归状态 |
| HF / PyTorch | 本项目使用 CPU FP32 作参考 | selected / full | 未接入 | Busbar 持有，串行分支 |
| vLLM Metal | 独立 Python 3.12 Mac 环境 | full | 未接入 | 引擎管理分页前缀缓存 |

`auto` 会选择该后端已经实现的读出方式。只有词表模式支持 selected/full 对照；显式请求不支持的组合会报错。MLX 选择头使用同一个底座加载器和缓存执行路径。

## 原生 Mac 安装与模型

需要 Apple Silicon、macOS 15+、Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。首次运行会从 Hugging Face 下载约 1.2 GB 的模型权重；权重保存在用户的 Hugging Face 缓存中，不进入本仓库。

```bash
git clone https://github.com/CCpro10/Busbar.git
cd Busbar
uv sync --locked --extra mlx --extra dev
uv run busbar run examples/returns.json
```

默认固定模型 `Qwen/Qwen3-0.6B`、revision `c1899de289a04d12100db370d81485cdf75e47ca`，MLX 使用 FP16，关闭额外思考与文本生成。输入、模型权重和推理均在本机；下载模型需要网络。

运行较强的模型（均未量化，本机 48 GiB 内存已验证）：

```bash
uv run busbar run examples/returns.json --model Qwen/Qwen3.5-4B --dtype bfloat16
uv run busbar run examples/returns.json --model openbmb/MiniCPM5-2B --dtype bfloat16
```

这两个型号都内置了不可变 revision。MiniCPM5 下载约 5 GB；Qwen3.5 下载约 9.3 GB 的官方多模态 checkpoint，原生后端只加载文本模型，不提供视觉接口。其他型号必须显式指定 `--revision`；默认仍保留 0.6B，避免升级后自动触发大模型下载。

## vLLM Metal

使用独立 Python 3.12 环境，避免 vLLM Metal 原生扩展需要的 MLX 版本与原生后端依赖冲突：

```bash
bash scripts/setup-metal.sh
.venv-metal/bin/busbar run examples/returns.json \
  --backend vllm-metal --model openbmb/MiniCPM5-2B
.venv-metal/bin/busbar serve --backend vllm-metal \
  --model openbmb/MiniCPM5-2B --port 8787
```

安装脚本使用官方 vLLM 0.29.0 / vLLM Metal 0.29.0 wheels，完整依赖版本记录在 `requirements-metal.lock`；安装后执行依赖兼容性检查。默认 BF16，`--memory-fraction 0.35` 限制引擎缓存预算。后端通过只读 worker 扩展验证真实权重类型，避免当前插件保留 BF16 权重、但命令行声称 FP16 的情况；MLX 对照跑分也要指定 `--dtype bfloat16`。先停掉同端口的旧服务，再启动新后端。

此后端真实执行完整词表头，每题请求一个生成 token，同时读取所有候选 token 的原始 log probabilities，随后在候选内重新归一化。它们与原始 logits 相差一个共同常数，softmax 比例保持一致；响应以 `logit_space="vocabulary_logprobs"` 标注。`projection="auto"` 自动选择 `full`；显式要求 `selected` 会拒绝，不声称已实现候选行投影。

**MiniCPM5 的 Metal 缓存模式仍有限制：** 原始退货示例重复请求时，Score 最大概率变化约 0.122，最高分等级发生变化，严格数值门禁未通过。17 项 HTTP 功能流程可以完成；`mode="fresh"` 的两次实测结果一致，但会失去前缀复用收益。现阶段优先使用 Qwen3.5-4B＋MLX；不要把 MiniCPM5/Metal 的“可以运行”理解成已保证缓存命中前后的数值稳定。失败记录与复现见 [0.3 报告](mac-v03.md)。

**已知上游退出问题：** 当前 macOS 15.6.1 + PyTorch 2.13.0 环境在引擎退出清理时可能发生 `empty_host_cache()` 段错误；独立调用也能复现，上游 [#765](https://github.com/vllm-project/vllm-metal/pull/765) 记录了同类问题。推理与常驻服务可工作，暂不宣称干净的 worker 退出。没有修改依赖源码或屏蔽错误。

## 资源与支持范围

模型权重和运行中的 KV 缓存是两类不同资源。`--cache-mib` 只限制原生后端保留的前缀，不限制模型权重、批量分支、激活或 MLX 分配器缓存；`--batch-size` 控制每批路径数。选择头的一道 K 选一题会产生 K 条路径。

当前 MLX/HF 适配 dense、未量化的 Qwen2/3/3.5 和 Llama 架构；MiniCPM5 使用 Llama 适配。模型 revision 必须是完整不可变 SHA。支持一种架构不等于验证了该架构下所有 checkpoint；本机实际模型证据见 [0.3](mac-v03.md) 和 [0.4](mac-v04.md)。

Qwen3.5 混合了注意力缓存与递归状态，两者在独立分支中都必须复制。MLX-LM 固定到包含 GDN 归一化修复的提交，`A_log` 保持 FP32；FP32 参考加载在转换权重前提升精度。这些适配集中在后端文件，不散入业务 Runtime。

Linux vLLM/SGLang、量化、LoRA/底座微调、跨 HTTP 请求连续批处理、KV offload、分布式或跨进程恢复尚未实现。HF 是正确性参考路径；本项目没有据此宣称 GPU 服务吞吐。添加后端的契约与验收步骤见 [架构说明](design.md)。
