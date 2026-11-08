# LoRA 领域微调实验（Qwen2.5-0.5B）

> 本仓库数据为合成演示数据，与任何真实企业经营数据无关；文档只描述方法论与工程实现。

> **目的**：在 hunkon_rag 品牌知识语料上做领域微调，让模型回答品牌业务问题时更贴合知识库表述。
> **诚实标注**：本实验提供**完整可复现的脚本与数据流水线**，但**未在本机实际执行训练**——编写环境无 GPU、未安装深度学习运行时。在任意有 GPU 的环境按本 README 即可运行。

---

## 一、为什么做这个实验

大模型通用能力很强，但回答垂直领域问题时往往：
1. 不知道品牌专属事实（门店数、新品名、加盟政策）
2. 表述与真相源口径不一致（如把合成门店数写错）

RAG 可以解决「知识检索」，但**微调解决「表述风格与领域一致性」**——两者互补。本实验演示后者。

---

## 二、数据流水线

```
data/corpus/*.md（品牌知识语料）
        │  finetune/prepare_data.py
        ▼
finetune/data/train.jsonl（12 条）+ val.jsonl（2 条）
        │  finetune/train_lora.py
        ▼
finetune/output/lora_adapter（LoRA 权重）
        │  finetune/eval_lora.py
        ▼
finetune/output/eval_compare.md（微调前后对比）
```

### 数据格式

每条 `{instruction, input, output}`：
- instruction：根据知识库回答「XX」
- output：corpus 原文（领域正确表述）

---

## 三、训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 基础模型 | Qwen/Qwen2.5-0.5B | 0.5B 小模型，单卡可训 |
| LoRA rank | 8 | 低秩维度 |
| LoRA alpha | 16 | 缩放因子 |
| target_modules | q/k/v/o_proj | 注意力投影层 |
| 学习率 | 2e-4 | LoRA 常用学习率 |
| epochs | 3 | 小数据避免过拟合 |
| max_length | 512 | 截断长度 |

---

## 四、运行步骤（需 GPU 环境）

```bash
# 1. 安装依赖
pip install torch transformers peft datasets accelerate

# 2. 准备数据（已生成，可重新跑）
python finetune/prepare_data.py

# 3. 训练
python finetune/train_lora.py --model Qwen/Qwen2.5-0.5B --epochs 3

# 4. 评估对比
python finetune/eval_lora.py --base-model Qwen/Qwen2.5-0.5B --adapter finetune/output/lora_adapter
```

---

## 五、预期效果与评估维度

微调后预期：
- 回答更贴合知识库表述（门店数 12 家、四款新品名、加盟条件等）
- 减少通用模型的「自由发挥」倾向

评估维度（`eval_compare.md`）：
- **事实正确性**：是否说出正确门店数/新品名
- **表述一致性**：是否使用知识库口径
- **简洁性**：是否避免无关展开

> 更严格的评估：可复用 hunkon_rag 的 golden set（`data/eval_golden.json`）做召回/忠实度评测，对比微调前后 faithfulness_rate。

---

## 六、诚实边界

- 本机未实际训练（无 GPU）；`train_loss.json`、`eval_compare.md` 的微调后部分**待有 GPU 环境后补充真实数据**。
- 0.5B 模型效果有限，生产级微调应使用更大模型（7B/14B）或更充分的数据。
- 本实验定位是「理解 LoRA 微调全流程」的工程演示，不是生产级模型训练。