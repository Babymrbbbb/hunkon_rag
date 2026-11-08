# hunkon_rag · LLMOps 设计（评测门禁 + 运行时监控）

> 本文档说明 hunkon_rag 的 LLMOps 落地形态：**评测门禁（离线可复现）+ 运行时指标 + 监控看板**。
> **诚实标注**：这是 LLMOps 的**核心雏形**，不是完整生产级流水线——暂缺模型版本管理（MLflow）、模型 CI/CD（GitHub Actions 级）、A/B 测试平台、生产监控告警（Prometheus/Grafana 级）。本文档对应「从 0 到 1 建立质量闭环」的架构师叙事。

---

## 一、为什么需要 LLMOps

大模型应用与传统软件的本质差异：**模型输出不可控，且依赖上游 API 的稳定性**。没有 LLMOps，AI 应用会陷入三个经典问题：

1. **改坏了不知道**：改了检索逻辑/语料/参数，问答质量下降，但没有门禁拦截。
2. **上游抖动无感知**：模型 API 限流（429）、故障（503）时，用户看到报错，但不知道是模型问题还是应用问题。
3. **效果无法度量**：没有 golden set 和回归，无法回答「这版比上版好还是差」。

LLMOps 的目标：**把 AI 应用的质量从「感觉」变成「可度量、可回归、可监控」**。

---

## 二、流水线全景

```
                    ┌─────────────────────────────────────────────┐
                    │                 hunkon_rag                     │
                    │                                             │
  语料/真相源 ──▶ 索引构建 ──▶ RAG 检索 ──▶ LLM 生成 ──▶ 答案+引用  │
                    │     │              │                           │
                    │     ▼              ▼                           │
                    │  评测门禁       运行时指标                     │
                    │  (golden set)   (metrics.py)                 │
                    │     │              │                           │
                    │     ▼              ▼                           │
                    │  eval_cli.py    /metrics 端点                │
                    │  (CI 可拦截)     │                           │
                    │     │              ▼                           │
                    │     ▼        监控看板 monitor.html            │
                    │  reports/*.md  (评测历史+实时KPI)             │
                    └─────────────────────────────────────────────┘
```

对应 LLMOps 四层：

| 层 | 本仓库实现 | 生产级补充（诚实标注） |
|---|---|---|
| **评测** | `data/eval_golden.json`（8 题）+ `app/eval.py`（召回/忠实度） | 更大 golden set、LLM-as-Judge |
| **门禁** | `app/eval_cli.py`：阈值拦截、非零退出码、报告落盘 | GitHub Actions CI 集成 |
| **可观测** | `app/metrics.py` 进程内指标 + `/metrics` 端点 | Prometheus 采集、Grafana 看板 |
| **监控** | `monitor.html` 读取 `/metrics` + `reports/history.json` | 告警规则、SLO 追踪 |

---

## 三、评测门禁（CI 可拦截）

### 3.1 评测集

`data/eval_golden.json` 覆盖 8 个核心业务场景（门店/新品/冰面包调研/包材/竞品/会员/加盟/招牌），每题含 `expected_doc`（期望召回文档）和 `expected_hint`（期望事实），用于计算：

- **召回命中率**：期望文档是否出现在 top-k 召回内（验证检索与切分正确）
- **忠实度**：期望事实是否出现在答案或召回的原文中（验证生成有依据）

### 3.2 命令行门禁

```bash
# 跑全部用例，默认阈值 0.95（与 CI 一致；8 用例下等价于要求 8/8），生成报告 + 写 history
python -m app.eval_cli --mock

# 自定义阈值（CI 可设更严）
python -m app.eval_cli --mock --threshold 0.8

# 输出 JSON（供 CI 解析）
python -m app.eval_cli --mock --json
```

**失败拦截**：召回命中率低于阈值时，进程以**退出码 2** 退出。CI 管道（如 GitHub Actions）可用 `if [ $? -ne 0 ]; then exit 1; fi` 阻断合并——这是「评测门禁」的工程落点。

**可复现性**：`--mock` 强制使用离线 MockProvider + 确定性 rule 编译聚合层，**不受上游 LLM 限流影响**，CI 随时可跑、结果稳定。

### 3.3 产物

- `reports/eval_report_<时间戳>.md`：每次评测的 Markdown 报告（含逐题明细）
- `reports/history.json`：时间序列历史，供监控看板展示趋势

---

## 四、运行时可观测

### 4.1 进程内指标

`app/metrics.py` 用线程安全计数器收集：

- `requests`：总请求数
- `chat_requests`：问答请求数
- `latencies`：每次问答延迟（ms）
- `total_tokens`：累计 Token 消耗

### 4.2 指标端点

```
GET /metrics
```

返回 JSON（需 X-API-Key）：

```json
{
  "requests": 128,
  "chat_requests": 96,
  "avg_latency_ms": 812.5,
  "total_tokens": 45231
}
```

---

## 五、监控看板

`/monitor`（免鉴权，深色主题）提供零依赖监控页：

- **实时 KPI**：总请求 / 问答请求 / 平均延迟 / Token 消耗（每 5 秒刷新）
- **评测历史**：读取 `reports/history.json`，展示每次评测的召回率、忠实度、通过状态与趋势条

看板设计为**架构展示**：证明「可观测性」不是空话，而是有真实数据端点 + 前端呈现。

---

## 六、与生产级 LLMOps 的差距（诚实边界）

| 能力 | 本仓库 | 生产级 | 说明 |
|---|---|---|---|
| 模型版本管理 | ❌ 无 | MLflow / DVC | 模型、prompt、参数未版本化 |
| CI/CD 集成 | ⚠️ 有 CLI 门禁，未接 CI | GitHub Actions / GitLab CI | eval_cli 已具备被 CI 调用的接口 |
| 在线评测 | ❌ 无 | 影子流量 / LLM-Judge | 仅离线 golden set |
| 告警 | ❌ 无 | Prometheus + Alertmanager | 看板不主动告警 |
| A/B 测试 | ❌ 无 | 流量切分平台 | 无多版本对比 |
| 成本治理 | ⚠️ mock/real 切换 | 配额审计 / 预算告警 | 有 provider 抽象，无预算控制 |

> 这份差距清单明确区分「已具备雏形」与「生产级缺失项」，并给出补齐顺序，便于后续迭代。

---

## 七、架构师视角：为什么这样设计

1. **评测先行**：没有 golden set 的 AI 应用，任何改动都无法判断好坏——评测是 LLMOps 的地基。
2. **门禁可执行**：不是「跑一下看看」，而是**失败即非零退出**，让质量卡点在流程上生效。
3. **可观测先于告警**：先有数据端点（/metrics、history.json），才有后续接 Prometheus/Grafana 的可能——架构上不堵路。
4. **诚实标注边界**：demo 到生产之间有明确的差距清单，不把「雏形」说成「生产级」，这是架构师的基本信用。

---

*本文档对应「LLMOps 雏形」实现说明。配套代码：`app/eval_cli.py`、`app/static/monitor.html`、`app/main.py`（/monitor、/reports/history.json 路由）。*