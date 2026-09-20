# 0.3：现代小模型与项目对照

实测日期：2026-09-21；Apple M4 Pro，48 GiB，macOS 15.6.1。
结论和适用范围见 [完整报告](../../../docs/mac-v03.md)。此目录保存逐条预测、计时样本、
数值检查和 HTTP 验收；模型权重、虚拟环境和 WANLI 原始文本不入库。

## 固定来源

- SemIf：`ca3ba65f142967030ecb453346e94d6f476a69df`；评分函数、提示词和数据构建器不修改。
- NanoJev：`618cea6d906d54e128360786d12f703fff2b1245`；使用原始模型类、token 构建器和训练权重。
- NanoJev checkpoint：`047b927b30882a1138fc504821b82ac145a4b81a`，根目录 `best.safetensors`，
  当前为游戏 SFT checkpoint。MPS 移植只涉及加载、设备和执行编排，不代表官方 CUDA 延迟。
- 模型、数据与文件校验值另见 `inputs.json`、`SHA256SUMS`。
- 原生对比统一用 `benchmarks/requirements-compare.lock`；MLX-LM 固定 Git 提交。
  Metal 使用独立 Python 3.12 和仓库根目录 `requirements-metal.lock`。

## 复现

以下从 Busbar 根目录执行。所有结果使用新路径，脚本拒绝覆盖已有记录。

```bash
git clone https://github.com/TheoLeeCJ/SemIf.git local-results/SemIf
git -C local-results/SemIf checkout ca3ba65f142967030ecb453346e94d6f476a69df
uv venv --python 3.13 local-results/venv-compare
uv pip install --python local-results/venv-compare/bin/python -r benchmarks/requirements-compare.lock

curl -L --fail \
  https://huggingface.co/datasets/alisawuffles/WANLI/resolve/61c95318fd71c55b6ba355d76253254615f387ec/test.jsonl \
  -o local-results/wanli-test.jsonl
shasum -a 256 local-results/wanli-test.jsonl
# 必须等于 4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd。
local-results/venv-compare/bin/python local-results/SemIf/benchmarks/build_wanli.py \
  --source local-results/wanli-test.jsonl \
  --selection local-results/SemIf/benchmarks/manifests/source-selection.jsonl \
  --output local-results/wanli256.jsonl

local-results/venv-compare/bin/busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --model Qwen/Qwen3.5-4B --dtype bfloat16 --output local-results/new-busbar-authored.json
local-results/venv-compare/bin/python benchmarks/compare_projects.py \
  --project semif --source-root local-results/SemIf \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/semif-authored144.jsonl --output local-results/new-semif-authored.json
# 将 project 改成 busbar-semif-prompt，得到完全相同提示词的 Busbar 对照。
# 将 input 改成 local-results/wanli256.jsonl，运行外部 256 题。

local-results/venv-compare/bin/python benchmarks/compare_projects.py \
  --project semif --task speed --source-root local-results/SemIf \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input local-results/SemIf/benchmarks/data/shape777.jsonl \
  --repeats 3 --output local-results/new-speed.json
```

速度脚本固定取 `shape777` 的第一个完整共享 state（21 题），不是跑满 777 题；七种路径
共用同一份已驻留模型和 tokenizer，随机轮换三轮，保留原始答案。含编码，不含模型加载。
冷缓存包含一次公共 state prefill；暖缓存预先做好 state prefill，不缓存问题的 KV。
SemIf 按原实现 padding 成一批，Busbar 按后缀长度分桶；两者都是实际生产代码。

NanoJev 的复现需要固定源码和 checkpoint：

```bash
git clone https://github.com/TianyuCodings/NanoJev.git local-results/NanoJev
git -C local-results/NanoJev checkout 618cea6d906d54e128360786d12f703fff2b1245
local-results/venv-compare/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("C-Tianyu/NanoJev", revision="047b927b30882a1138fc504821b82ac145a4b81a", allow_patterns=["config.json", "best.safetensors", "backbone_config/*", "tokenizer/*"]))'
# 将输出的实际 checkpoint 目录代入 --checkpoint。
local-results/venv-compare/bin/python benchmarks/compare_projects.py \
  --project nanojev --source-root local-results/NanoJev \
  --revision 047b927b30882a1138fc504821b82ac145a4b81a \
  --checkpoint /absolute/path/to/checkpoint --device mps --nano-autocast \
  --input benchmarks/data/semif-authored144.jsonl --output local-results/new-nano-authored.json
```

`nanojev-bf16-*` 是正文使用的 BF16 autocast 结果，参数存储为 FP32，与上游 CUDA 的精度策略
一致，但设备不同。`nanojev-authored144.json` 是早期 FP32 MPS 诊断；它的源码 checkout 为
`71a513bb0163b5634467842b523ee0c0ed6fb1c7`，实际模型/预测两个文件与上述新版逐字节一致，
校验值在 `inputs.json`。CPU/MPS 诊断固定使用最前 8 题，不能扩展为 CUDA 全量一致性证明。

Metal 质量、性能与真实 HTTP 复现：

```bash
bash scripts/setup-metal.sh
.venv-metal/bin/busbar evaluate benchmarks/data/semif-authored144.jsonl \
  --backend vllm-metal --model openbmb/MiniCPM5-2B --output local-results/new-metal-authored.json
.venv-metal/bin/busbar benchmark --backend vllm-metal --model openbmb/MiniCPM5-2B \
  --context-tokens 2048 --questions 8 --repeats 3 --output local-results/new-metal-speed.json
.venv-metal/bin/busbar serve --backend vllm-metal --model openbmb/MiniCPM5-2B --port 8787
# 另一个终端：
.venv-metal/bin/python examples/http_smoke.py --output local-results/new-http.json
```

MiniCPM5 的该 HTTP 测试**预期会以非零退出**：协议流程完成，但 APC 重复请求数值门禁失败。
脚本仍保存全部响应和失败检查，不会将其标成通过；原有 0.02 概率阈值和选项一致要求未放宽。
`http-minicpm5-fresh.json` 是额外的无 APC 读取重复请求诊断，不替代缓存模式失败记录。

`comparison.json` 可用 `python benchmarks/summarize_comparison.py benchmarks/results/mac-v03
--output local-results/new-summary.json` 重算，脚本会校验逐条 ID、gold、候选顺序、模型 revision、
提示词哈希和 token 数。`SHA256SUMS` 用 `shasum -a 256 -c SHA256SUMS` 校验。

WANLI 作者与数据来源：[alisawuffles/WANLI](https://huggingface.co/datasets/alisawuffles/WANLI)，
许可 CC-BY-4.0；保留原题 ID、来源版本与标签 provenance。SemIf authored144 和 shape777
来自该项目 MIT 许可样例，许可文件已在 `benchmarks/data/SemIf-LICENSE` 保留。
400 题合并分数是本次实验的汇总，不是 Typed Decision Bench 或 Jev 官方榜单成绩。
