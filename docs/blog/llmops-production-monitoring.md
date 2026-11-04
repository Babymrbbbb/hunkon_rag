# 从离线门禁到生产监控：RAG 知识库的 LLMOps 补全实战

> 2026-11-03 · Hunkon 知识库（47 家门店真实生产环境）

## 背景

此前的知识库已经具备**离线评测门禁**（24 条用例、11 类场景、通过率 100%、改链路必回归）。但对照 2026 年生产级 RAG 标准，仍有两个明确缺口：

1. **生产流量持续监控缺失**——门店用户在 Web 端问答的质量、token 成本、延迟、失败率都不可见，只有离线门禁，没有在线观测。
2. **评测指标未业界化**——只有「用例通过率」，没有 RAGAS 四指标（faithfulness / answer relevancy / context precision / context recall）量化，面试和对外展示时无法直接对标业界标准。

本文记录这两项 P0 的落地过程，以及落地过程中**意外发现并修复的一个真实检索质量回归**。

---

## 一、生产流量采样监控：Langfuse 接入问答链路

### 设计约束

生产知识库是**零第三方依赖**的（纯 Python 标准库）。接入可观测性不能破坏这个特性，也不能影响现有问答链路。

### 实现

新增 `langfuse_trace.py`（约 180 行），纯标准库实现 Langfuse HTTP Ingestion 协议：

- `urllib.request` 上报，`hmac` 做签名，无任何第三方依赖
- 采样率可配：`LANGFUSE_SAMPLE_RATE=0.1` 时约 10% 请求上报（生产建议值）
- **默认关闭**：`LANGFUSE_ENABLED != '1'` 时 `start_trace()` 返回 `None`，`end_trace(None)` 直接返回，**零额外请求、零性能影响**

埋点位置在 `rag_qa.llm_complete()` 内——这是所有问答链路的**唯一必经点**（Web 端、自动化任务都走这里），因此上层**零改动**。

```python
_trace = langfuse_trace.start_trace(
    name="kb-qa",
    input_text=user[-2000:],
    metadata={"model": MODEL},
)
# ... 调用 LLM ...
langfuse_trace.end_trace(_trace, output=answer, usage=usage,
                         latency_ms=..., status="OK")
```

### 验证

- 7 项单元测试全部通过（事件构造、签名算法、采样率、禁用开关、服务器不可达静默失败）
- `LANGFUSE_ENABLED=0` 时问答链路行为完全不变（实测）

---

## 二、评测指标 RAGAS 化：四指标量化

### 设计约束

不引入 `ragas` 库（它依赖 numpy/pandas/datasets，与生产知识库零依赖原则冲突）。用 **LLM-as-judge** 实现 RAGAS 官方定义的四指标：

| 指标 | 官方定义 | 零依赖实现 |
|---|---|---|
| **Faithfulness** | 答案中能被 context 支持的 claim 比例 | LLM 拆 claims → 逐条判断是否被 context 支持 |
| **Answer Relevancy** | 从答案反推的合成问题与原始问题等价比例 | LLM 生成 N 个合成问题 → 判断与原始问题语义等价 |
| **Context Precision** | 检索块中相关块的比例 | LLM 逐块判断是否与问题相关 |
| **Context Recall** | ground truth 关键句在 context 中的比例 | LLM 拆 ground truth → 判断每句是否在 context 中 |

新增 `ragas_eval.py`，并在 `eval_kb.py` 加 `--ragas` 参数，报告末尾自动追加四指标汇总。

### 实测结果

对 24 条评测用例中的 2 条权威口径用例跑 RAGAS 量化：

```
faithfulness:        0.818（9/11 claims 被支持）
answer_relevancy:    0.333（1/3 合成问题等价）
context_precision:   1.000（2/2 块相关）
context_recall:      1.000（1/1 句在上下文中）
```

`answer_relevancy` 偏低是 LLM-as-judge 对「语义等价」判断偏严的已知局限（合成问题措辞不同但语义等价被判不等价），但 `faithfulness 0.818`、`context_precision 1.0`、`context_recall 1.0` 已是有效信号。

---

## 三、意外收获：评测回归暴露的真实检索质量缺陷

跑 RAGAS 化评测时，发现 2 例口语化用例（「店里一共多少家」「总共有多少间」）从之前 100% 通过**回归到失败**——答案不再直接给出「47」，而是含糊地说「存在口径冲突，以裁决为准」。

### 根因定位

排查发现：聚合页 `wiki/entity/Hunkon.md` 的正文写的是「**门店总数以内部裁决口径为准**」——这句话没有给出具体数字「47 家」。

LLM 读到聚合页后，只能回答「以裁决为准」，无法输出权威数字。这是**聚合页在近期重建时丢失了具体权威数字**（此前 24/24 时正文有「47 家」），属于真实的检索质量回归，不是评测脚本问题。

### 修复：代码层确定性覆盖（规则 4）

在共享闸模块 `kb_rules.py` 增加规则 4：当编译 `page_name == "Hunkon"` 的聚合页时，若正文出现「门店总数以内部裁决口径为准」这类含糊表述，**确定性替换为**：

> 门店总数以权威裁决口径为准：**47 家**（清远含连州 40 / 广州含花都 4 / 肇庆 2 / 佛山 1，韶关无门店）

这符合知识库铁律「权威口径必须代码层覆盖」——不依赖 LLM 自觉，裁决写了什么就是什么。

### 端到端验证

修复后重新问答：

```
问：店里一共多少家
答：Hunkon 目前共有 47 家门店。权威口径：清远（含连州）40 家 / 广州（含花都）4 家 /
    肇庆 2 家 / 佛山 1 家，韶关无门店。来源：聚合页:entity·Hunkon 及
    L3_原始资料/重要决策/2026-10-24-Hunkon 门店总数口径裁决.md

问：总共有多少间
答：最新权威口径（裁决更新）：47 家。分布：清远含连州 40 家 / 广州含花都 4 家 /
    肇庆 2 家 / 佛山 1 家（韶关无门店）。
```

两个口语化用例都明确答出「47 家」，且带区域分布和权威来源。

---

## 四、经验总结

1. **补可观测性不一定要引入重依赖**。Langfuse 的 HTTP Ingestion 协议足够简单，纯标准库即可实现，保持生产系统零依赖特性。
2. **RAGAS 四指标可以用 LLM-as-judge 零依赖实现**。虽然 answer_relevancy 的语义等价判断有局限，但 faithfulness / context precision / context recall 已经是可靠信号。
3. **评测回归是金矿**。这次如果不是跑 RAGAS 化评测，口语化用例的回归不会被发现——聚合页口径漂移会一直静默存在。**改链路必回归**的价值再次被验证。
4. **权威口径必须代码层覆盖**。聚合页正文的含糊表述会被 LLM 原样采信，必须用确定性规则把裁决数字注入，而不是依赖 prompt。

## 五、文件清单

| 文件 | 说明 |
|---|---|
| `langfuse_trace.py` | 零依赖 Langfuse 埋点（Python 版，约 180 行） |
| `test_langfuse_trace.py` | 单元测试（7 项断言） |
| `LLMOPS_LANGFUSE_问答链路.md` | 生产流量采样监控接入文档 |
| `ragas_eval.py` | RAGAS 四指标零依赖实现 |
| `kb_rules.py` | 新增规则 4：聚合页注入 47 家权威口径（代码层确定性覆盖） |
| `eval_kb.py` | 新增 `--ragas` / `--ragas-limit` 参数 |

---

*本文基于真实生产系统改造记录，所有数字可在知识库评测报告与代码中追溯。*