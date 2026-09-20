# 快捷接入与可训练选择头

Busbar 提供两种读出方式，共用 ContextSnapshot、Boolean、Choice、Score，以及相同的 Python / HTTP API。应用可以把工单分流、条件判断、风险等级、检索候选筛选等功能表达为“上下文＋问题＋候选描述”。模型只返回决策数据，具体业务操作由应用执行。

| | 快捷模式 | 可训练选择头 |
|---|---|---|
| 模型输入 | 一条路径包含问题和所有选项 | 每个候选各有一条路径，复用上下文前缀 |
| 得分来源 | 已有词表中 A/B/C/D 等标签对应的行 | 新的共享 LayerNorm＋线性标量层 |
| 是否需要标注训练 | 不需要 | 需要；本版冻结底座，只训练头 |
| 候选数量 | 每题 2–16 个 | 每题 2–16 个；参数数量与候选数量无关 |
| 推理后端 | MLX / HF / vLLM Metal | 原生 MLX；其他后端显式拒绝加载头 |
| 默认投影 | MLX/HF 为 selected，Metal 为 full | head；不会执行词表投影 |
| 适合 | 已能用指令完成的任务、快速验证 | 有独立标注数据、需要学习任务规则或偏好的场景 |

这里采用 NanoJev 的候选描述＋共享评分头思路，但并非 NanoJev checkpoint 的兼容加载器。本版没有候选集合 attention，也没有 LoRA、全参数微调或软标签蒸馏。Boolean 统一编码为两个候选，区别于 NanoJev 的单路径 Boolean 实现。实现参考其[固定版本训练代码](https://github.com/TianyuCodings/NanoJev/blob/618cea6d906d54e128360786d12f703fff2b1245/scripts/train_toy_decisions.py)，MLX 训练使用[官方自动求导接口](https://ml-explore.github.io/mlx/build/html/python/nn.html)。

## 一句话选项怎样变成分数

以客服分流为例，上下文是客户说“我付款一次，银行卡却扣了两次”，问题是“应交给哪个团队”，候选描述是“配送查询”“退货退款”“支付账单”。

对每个候选 i，编码器拼接以下内容并追加 EOS：

```text
Instructions: 应用指令
State: 稳定序列化的 JSON
Question type: choice
Question: 应交给哪个团队？
Candidate: 支付账单
Decision:
```

上下文和问题/候选分段分词；训练、缓存推理和 fresh 推理使用完全相同的 token 拼接规则。底座 Transformer 在最后一个 token 处给出隐藏向量 hᵢ。它包含上下文、问题和当前候选的联合信息。选择头计算：

```text
zᵢ = wᵀ LayerNorm(hᵢ)
pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
训练损失 = -log(p正确候选)，训练时 T = 1
```

一句话选项先经过完整的 Transformer 编码，再被读成一个标量；没有把整句话硬塞成一个词表 token。w 在所有候选之间共享，因此换一批候选或从 3 个选项改成 5 个，不需要改变头的形状。选项 ID 和位置不进入候选文本：调整排列或改业务 ID，只改变结果映射。增加/删除一个候选时，其余候选的原始分数不变，softmax 概率会随候选集合重新归一化。

LayerNorm 的可训练缩放、偏移，加上线性权重，共有 3D 个参数；D 是隐藏维数。标量层没有偏置，因为给所有候选加同一个常数不会改变 softmax。底座可以采用 FP32、FP16 或 BF16，头始终以 FP32 计算。

新头的随机参数最初没有学到“隐藏向量的哪个方向表示正确候选”。底座已有的语言知识可以提供有用特征，但不会自动替随机 w 规定评分含义。训练通过交叉熵更新头，底座权重保持冻结。这也是为什么现有指令模型已经能做好任务时，快捷模式可能更好：它保留了已经训练好的词表读出能力。训练头能否胜出需要独立测试，不能由结构直接推出。

## 训练自己的功能

先安装原生 Mac 环境 `uv sync --extra mlx --extra dev`。每行一个 JSON 对象，三种标签分别为 **JSON 布尔值、选项 ID、已声明的数值等级**，都不是选项位置。

```json
{"id":"ticket-1-route","group_id":"ticket-1","context":{"state":{"message":"我的银行卡重复扣款了"},"instructions":"根据客户诉求选择团队"},"question":{"type":"choice","question":"哪个团队应处理？","options":[{"id":"delivery","description":"配送与物流"},{"id":"billing","description":"支付、账单与重复扣款"}]},"label":"billing"}
```

Boolean 的 `label` 写 `true` / `false`；Score 的 `label` 写如 `2`，且必须在 `question.levels` 中。Score 训练是等级分类，推理的 `selected` 是最可能等级，`value` 仍是各等级数值的概率期望。它不是独立的连续回归模型。概率未校准，不能直接当成业务正确率。

同一个原始工单及其所有衍生问题使用同一个 `group_id`。先按工单/用户/来源划分 train、validation、test，再派生问题，避免同一事件进入多个集合。程序检查 ID、group 和精确语义输入重叠；无法自动识别改写文本中的语义泄漏，划分质量仍由数据作者负责。

仓库的[客服样例](../examples/trainable_support/)提供 96 条训练、36 条验证、48 条测试记录，分别来自 32、12、16 个不重叠工单。包含三个决策类型和 2–4 个动态 Choice 候选。模板在训练前固定，测试集不参与选权重或调参；这是小型流程验证样例，不是通用能力榜单。

```bash
# 可重复生成示例；不是调用大模型生成标签。
uv run python examples/trainable_support/make_dataset.py

# 默认 0.6B、FP32。模型先提取候选特征，再只训练小头。
uv run busbar train-head examples/trainable_support/train.jsonl \
  --validation examples/trainable_support/validation.jsonl \
  --output models/heads/support-v1 --epochs 40

# 较大模型也走相同流程；以下命令与上面二选一或使用不同输出目录。
uv run busbar train-head examples/trainable_support/train.jsonl \
  --validation examples/trainable_support/validation.jsonl \
  --model openbmb/MiniCPM5-2B --dtype float32 \
  --output models/heads/support-minicpm5-v1 --epochs 40
```

Qwen3.5-4B 同样已经实测；替换 `--model Qwen/Qwen3.5-4B` 即可。内置模型固定不可变 revision；其他型号必须给出完整 SHA，并受现有 MLX 架构支持范围约束。FP32 训练默认保留更稳定的缓存数值，大模型内存不够时可在训练前选择 BF16；加载时必须保持相同精度。

`--batch-size` 控制底座候选路径批大小，`--train-batch-size` 控制头训练批大小，二者不同。`--feature-cache-mib` 默认 256，限制候选特征及堆叠副本的预估空间，**不包括底座权重、临时计算和 KV 缓存**。超长输入、特征预算不足、非法标签、空数据、重复 ID 或数据划分重叠会报错，不静默截断/丢弃记录。所有候选长度在执行第一条训练前向前统一检查。

## 验证、服务和接入应用

```bash
# 同一份独立测试集比较快捷模式和选择头；精度保持一致。
uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --dtype float32 --output local-results/quick.json
uv run busbar evaluate examples/trainable_support/test.jsonl --format native \
  --head models/heads/support-v1 --output local-results/head.json

uv run busbar serve --head models/heads/support-v1 --port 8787
uv run busbar run examples/returns.json --head models/heads/support-v1
```

`returns.json` 展示 API 兼容性；客服样例训练出来的头不代表已经学会退货政策。真实业务应换成自己的上下文和问题，用该业务的独立测试集验收。`--head` 自动读取底座模型、revision、dtype；显式传入不一致的值会报错。

```python
from busbar import ContextSpec, DecisionRequest, Runtime
from busbar.backends import load_backend

# 快捷模式：Runtime(load_backend("mlx"))
runtime = Runtime(load_backend("mlx", head="models/heads/support-v1"))
snapshot = runtime.compile_context(ContextSpec(state={"message": "我的银行卡重复扣款了"}))
result = runtime.decide(
    DecisionRequest(
        snapshot_id=snapshot.snapshot.id,
        questions={
            "needs_billing": {
                "type": "boolean",
                "question": "这条诉求是否需要支付或账单团队处理？",
            }
        },
    )
)
print(result.decisions["needs_billing"].probabilities)
```

Python 和 HTTP 均可在一次请求中混合 1–64 个 Boolean、Choice、Score 问题。`projection="auto"` 在该后端选择 `head`；显式 `selected` / `full` 会拒绝，防止误用词表结果。响应标记 `trained_candidate_logits`、候选路径数和未执行词表投影。模型仍可能保留词表权重，尤其输入嵌入与输出权重共享时；这里省掉的是词表乘法，并不承诺把整个词表权重从内存移除。

K 个候选需要 K 条 Transformer 路径，因此它不天然比快捷模式快。共享前缀只 prefill 一次，后续分支按长度组批；每个候选都计入计算 token 和复用 token 数。`busbar benchmark --head ...` 可以测 fresh/cached、顺序/批量，`profile` 的词表头拆解不适用于选择头。

## 保存、继续训练和回退

一个版本目录包括 `head.safetensors`、`training.json`、最后写入的 `manifest.json`。训练记录包含随机种子、超参数、每轮损失、验证集选出的轮次、参数变化、数据 SHA256 和划分记录；不保存底座的重复副本。第一次训练若没有超过随机初始头的验证 NLL，会明确失败，不发布随机头。

输出目录只能新建。写入中断或训练失败会留下不完整目录，加载器会拒绝；排查后换新目录重试，旧版本仍可用。加载检查固定文件名、文件哈希、头结构、每个张量的名称/形状/FP32/有限值、底座 revision 和 dtype。运行时 snapshot 身份包含头版本和编码器版本，两个头不能共用旧快照。

```bash
uv run busbar train-head path/to/new-train.jsonl \
  --validation path/to/new-validation.jsonl \
  --init-head models/heads/support-v1 \
  --output models/heads/support-v2

# 停止旧服务后切换；回退只需重新指定 v1。
uv run busbar serve --head models/heads/support-v2 --port 8787
```

`--init-head` 是从旧参数继续训练，优化器重新初始化，不是中断位置的精确恢复。新版本继承旧版本的训练/验证集记录：旧训练数据不能被当成新验证数据，测试时也会检查整个继承链。原生评测默认拒绝训练或验证集重叠；只有检查拟合情况时才显式使用 `--allow-training-data`，这类结果不属于独立测试。

若继续训练的验证损失没有优于旧头，会保留旧参数，报告 `best_epoch=0` 和 `parameter_delta_l2=0`；训练过程仍完整记录，不把一次训练操作自动等同于效果提升。

版本是普通目录，可以通过文件管理器列出、归档或删除。删除磁盘版本不会卸载已经驻留的模型，停止其服务才能释放内存；仍需加载/回退的版本不要删除。服务不热替换参数，切换版本时新建 Runtime 或重启进程，客户端重建 snapshot。上下文的创建、修改、列出/过滤、删除、过期和缓存清理仍沿用原有 API。

实际训练质量、快捷模式对照、SemIf/NanoJev 对照和完整验证记录见 [0.4 本机报告](mac-v04.md)。
