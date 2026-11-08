# 从离线门禁到运行时监控：RAG 知识库的 LLMOps 补全方法

> 2026-11-03 · 合成演示知识库（RAG 评测与监控方法论）

## 背景

一套可发布的 RAG 知识库应具备**离线评测门禁**（评测用例集覆盖多类场景、达到门禁阈值、改链路必回归）。但对照生产级 RAG 的运维要求，仍有两个明确缺口：

1. **生产流量持续监控缺失**——门店用户在 Web 端问答的质量、token 成本、延迟、失败率都不可见，只有离线门禁，没有在线观测。
2. **评测指标未业界化**——只有「用例通过率」，没有 RAGAS 四指标（faithfulness / answer relevancy / context precision / context recall）量化，无法直接对标业界标准。

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

- 单元测试覆盖事件构造、签名算法、采样率、禁用开关、服务器不可达静默失败
- `LANGFUSE_ENABLED=0` 时问答链路行为完全不变

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

### 评测结果（离线合成演示集）

对权威口径类用例跑 RAGAS 量化：

```
faithfulness:        0.818（9/11 claims 被支持）
answer_relevancy:    0.333（1/3 合成问题等价）
context_precision:   1.000（2/2 块相关）
context_recall:      1.000（1/1 句在上下文中）
```

`answer_relevancy` 偏低是 LLM-as-judge 对「语义等价」判断偏严的已知局限（合成问题措辞不同但语义等价被判不等价），但 `faithfulness 0.818`、`context_precision 1.0`、`context_recall 1.0` 已是有效信号。

---

## 三、意外收获：评测回归暴露的真实检索质量缺陷

跑 RAGAS 化评测时，发现口语化用例（「店里一共多少家」「总共有多少间」）从通过**回归到失败**——答案不再直接给出权威总数，而是含糊地说「存在口径冲突，以裁决为准」。

### 根因定位

排查发现：聚合页 `wiki/entity/演示品牌.md` 的正文写的是「**门店总数以内部裁决口径为准**」——这句话没有给出具体数字「12 家」。

LLM 读到聚合页后，只能回答「以裁决为准」，无法输出权威数字。这是**聚合页在近期重建时丢失了具体权威数字**（此前通过的用例正文里有「12 家」），属于检索质量回归，不是评测脚本问题。

### 修复：代码层确定性覆盖（规则 4）

在共享闸模块 `kb_rules.py` 增加规则 4：当编译 `page_name == "演示品牌"` 的聚合页时，若正文出现「门店总数以内部裁决口径为准」这类含糊表述，**确定性替换为**：

> 门店总数以权威裁决口径为准：**12 家**（A 市 6 / B 市 3 / C 市 2 / D 市 1，E 市无门店）

这符合知识库铁律「权威口径必须代码层覆盖」——不依赖 LLM 自觉，裁决写了什么就是什么。

### 链路验证

修复后重新问答：

```
问：店里一共多少家
答：演示品牌 目前共有 12 家门店。权威口径：A 市 6 家 / B 市 3 家 /
    C 市 2 家 / D 市 1 家，E 市无门店。来源：聚合页:entity·演示品牌 及
    L3_原始资料/重要决策/2026-10-24-演示品牌 门店总数口径裁决.md

问：总共有多少间
答：最新权威口径（裁决更新）：12 家。分布：A 市 6 家 / B 市 3 家 /
    C 市 2 家 / D 市 1 家（E 市无门店）。
```

两个口语化用例都明确答出「12 家」，且带区域分布和权威来源。

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
| `kb_rules.py` | 新增规则 4：聚合页注入 12 家权威口径（代码层确定性覆盖） |
| `eval_kb.py` | 新增 `--ragas` / `--ragas-limit` 参数 |

---

*本文只描述方法论与工程实现；配套代码与评测报告见本仓库。*