# 天鹅公主 AI 知识中台（RAG 服务）

> **在线体验**：公网部署完成后，在此填入演示地址（如 `https://rag.yourdomain.com`，`/` 为免鉴权演示页，`/docs` 为交互式文档）。
> 部署步骤见《公网部署操作手册》（本仓库 `docs/deploy.md` 或随附文档）。

> **定位**：面向「天鹅公主 SWAN NOBLE」**47 家门店烘焙连锁**的 AI 知识中台**落地方案与架构实现**。
> 由品牌运营负责人主导设计，目标是把分散在 SOP、新品资料、竞品情报、加盟政策中的知识，
> 收敛成一个「问即得、带出处、可审计」的检索增强生成（RAG）服务，直接服务一线门店与运营决策。

这不是一个脱离业务的"玩具 demo"——知识库的维度（品牌 / 门店网络 / 夏季新品 / 包材 / 会员 / 供应链 / 竞品金粒门 / 加盟 / 数字化）全部映射品牌真实经营切面；架构已用真实 LLM 端到端验证。**对外呈现的是"为真实业务设计的 AI 落地方案"，而非"练手项目"。**

---

## 一、业务背景与痛点

| 业务切面 | 现状痛点 | AI 中台要解决的 |
|---|---|---|
| 一线门店 | 新员工背 SOP / 产品知识靠口口相传，查标准慢、易走样 | 自然语言问答，答案带资料出处 |
| 新品上市 | 2026 夏季 4 款新品（黑金可颂/联名桃酥/荔枝玫瑰冻芝士/青提冷萃欧包）知识同步滞后 | 上市即入知识库，门店随问随答 |
| 竞品情报 | 金粒门等竞品动态分散在若干分析文档，无人能快速检索 | 竞品维度结构化、可追问 |
| 加盟 / 供应链 | 政策条款多、跨文档，加盟商咨询重复劳动高 | 政策问答 + 引用原文，降低人工成本 |
| 运营决策 | 市场/竞品分析散落，难沉淀复用 | 知识资产化，支撑复盘与决策 |

**量化目标（方案阶段，待落地验收）**：一线查标准 SOP 从「翻文档数分钟」降到「对话秒级」；竞品/政策类咨询的人工承接量目标下降 **30%+**；知识更新从「月度同步」提升到「随业务发生即入库」。

---

## 二、技术栈

| 层 | 选型 |
|---|---|
| Web 框架 | FastAPI（路由 / Depends 注入 / 全局鉴权 / 统一错误信封 / lifespan） |
| 数据模型 | SQLModel（SQLAlchemy + Pydantic v2） |
| 配置 | pydantic-settings（`RAG_` 前缀，env / `.env` 覆盖） |
| 存储 | SQLite + SQLModel（生产可换 pgvector） |
| LLM / 向量化 | **供应商解耦**：`LLMProvider` / `Embedder` 抽象；向量化三档 `mock`（哈希词袋，CI 用）/ `local`（本地 ONNX，真语义）/ `real`（OpenAI 兼容接口） |
| 检索 | 文本切分 + 向量余弦召回 top-k + 倒排 BM25-lite，RRF 融合 |
| 测试 / 评测 | pytest + golden-set 评测脚本 |
| 部署 | Docker 分层构建 + docker-compose |

### 向量化三档（2026-11-03 新增 local）

| provider | 实现 | 维度 | 用途 |
|---|---|---|---|
| `mock` | MD5 分词哈希词袋 | 256 | CI 与单元测试（确定性、零依赖、离线可复现） |
| `local` | 本地量化 ONNX（qwen3-embedding-0.6b） | 1024 | 本地/演示的真语义检索，CPU 推理，数据不出机 |
| `real` | OpenAI 兼容 `/embeddings` | 随模型 | 接云端向量服务 |

- 切换方式：`RAG_EMBEDDER_PROVIDER=local` + `RAG_EMBEDDER_LOCAL_DIR=<模型目录>`
- 依赖：`pip install -r requirements-local.txt`（CI 不装，CI 固定用 mock）
- **换 provider 会自动重建索引**：`seed.py` 把指纹（如 `local:1024`）写入 `Meta` 表，
  启动与评测时比对，不一致即重建——避免「旧库 256 维 + 新查询 1024 维」导致余弦恒为 0 的静默失效
- 模型缺失或依赖缺失时自动降级 mock，服务不中断
- 对照评测：`python -m app.eval_cli --mock --embedder local`（LLM/聚合页保持 mock，只换向量）

**真向量到底有没有用？** 门禁用的 8 条 golden 偏关键词型（问题里直接出现品牌名/产品名），
BM25 关键词路本就能命中，RRF 融合会把向量差异掩盖——mock 与 local 都是 8/8，看不出区别。
因此另设一套「语义探针」（`data/eval_semantic_probe.json`，12 条同义改写、几乎无字面重叠的问题）：

| provider | top1 命中率 | top3 命中率 |
|---|---|---|
| `mock`（哈希词袋） | 33.3% | 75.0% |
| `local`（qwen3-embedding-0.6b） | **75.0%** | **100.0%** |

```bash
python -m app.eval_probe                 # mock vs local 对照，输出 reports/probe_<时间戳>.md
python -m app.eval_probe --embedders mock --json
```

探针**不设门禁**（退出码恒 0）：CI 没有 613 MB 模型，local 必然降级为 mock，接门禁只会误报。
它只回答一个问题——「向量层值不值得开」。

---

## 三、架构图

### 业务视角：47 家门店全域数据整合 + AI 落地系统

```
┌─────────────────────────────────────────────────────────────────────┐
│                        门店一线 / 运营 / 决策者                        │
└───────────────┬─────────────────────────────────────────────────────┘
                │  自然语言查标准 · 查活动 · 查供应链 · 看经营分析
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    应用层：AI 嵌入门店日常分析工作                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────────┐ │
│  │  知识库检索  │ │ 经营分析看板 │ │ 会员运营助手 │ │ 竞品/行业研究   │ │
│  │  (RAG问答)  │ │ (次日自动)  │ │ (RFM+券)   │ │ (内容/自动化)   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────────┘ │
└───────────────┬─────────────────────────────────────────────────────┘
                │  AI 整合数据产出六大类应用
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AI 层：数据整合 → 智能应用                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  LLM×RAG 知识库   经营分析(66h→4h)   内容生产(21h→3h)         │  │
│  │  行业研究(2周→1.5天)  竞品拆解   看板自动化                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  底座：本地模型网关（10 池 16 模型）· 44 条自动化任务 · 44 个 AI 角色  │
│        执行验收分离 · 365 条教训库 · 年省约 5,800 工时（≈3.2 人力年） │
└───────────────┬─────────────────────────────────────────────────────┘
                │  统一数据口径（同一个数一个值）
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    数据层：全域数据整合底座                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │
│  │ ERP 进销存    │ │ 门店收银系统  │ │ 公域平台     │ │ 私域 SCRM  │  │
│  │              │ │              │ │ 抖音 / 美团  │ │            │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘  │
│  覆盖 47 家门店 · 服务高层 15 人 + 营运部 4 人 · 次日自动生成分析      │
└─────────────────────────────────────────────────────────────────────┘
```

> **读图方式**：最底层是 ERP→收银→公域→私域四条数据源，统一口径后形成 47 家门店一条数据链路；中间层由 AI 整合数据产出六大类应用；最上层把 AI 嵌入门店日常——员工自然语言提问，店长为验收方，使用统计 + 零使用预警闭环。**这不是几个独立项目，而是一套从数据到门店日常的全域整合系统。**

### C4 · 容器层（Container）

```
┌──────────────────────────────────────────────────────────────┐
│                    门店一线 / 运营 / 决策者                     │
└───────────────┬──────────────────────────────────────────────┘
                │  POST /ingest · POST /chat · GET /metrics · POST /eval
                │  Header: X-API-Key
                ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI 服务 (app/main.py)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ 路由 + 鉴权 │  │ 统一错误信封 │  │ lifespan    │  │ 指标收集  │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
│        │                  │                    │                 │
│        ▼                  ▼                    ▼                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  完整双轨管线 (app/rag.py)                                │   │
│  │  聚合层: wiki.search(question) → 命中聚合页优先            │   │
│  │  ingest: 切分 → 向量化 → 存 Chunk                          │   │
│  │  chat:   聚合页 + 混合检索(倒排+向量) → 拼 prompt → LLM     │   │
│  └───────────┬───────────────────────┬─────────────────────┘   │
│              ▼                       ▼                          │
│   ┌──────────────────┐      ┌────────────────────────────┐     │
│   │ LLMProvider 抽象  │      │ Embedder 抽象              │     │
│   │  Mock / OpenAI兼容│      │  Mock(哈希) / OpenAI兼容   │     │
│   └──────────────────┘      └────────────────────────────┘     │
│              │                       │                          │
│              ▼                       ▼                          │
│   ┌──────────────────────────────────────────────────────┐    │
│   │  wiki 层: data/wiki/index.json（聚合页索引）          │    │
│   │  SQLite: Document / Chunk(embedding JSON) / QA      │    │
│   └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 时序图（一次 /chat 请求）

```
Client → API网关(鉴权X-API-Key) → /chat
  /chat → wiki.search(question)            // 聚合层优先：命中聚合页
  /chat → Embedder.embed(question)          // 向量化查询
  /chat → Retriever.search(top_k, query_text) // 混合检索：倒排(BM25) + 向量(RRF融合)
  /chat → 组装 prompt(聚合页/召回原文 + 系统提示)
  /chat → LLMProvider.complete(prompt)     // Mock 或 真实 LLM
  /chat → 记录指标 / 落库 QA
  /chat → {answer, citations, latency_ms} → Client
```

---

## 四、目录结构

```
swan_rag/
├── app/
│   ├── main.py          # FastAPI 入口：路由 + 全局鉴权 + 统一错误信封 + lifespan
│   ├── config.py        # pydantic-settings（RAG_ 前缀）
│   ├── database.py      # 引擎 / init_db / get_session（Depends 注入）
│   ├── models.py        # 表模型：Document / Chunk / QA
│   ├── schemas.py       # 请求/响应模型（入参出参分离）
│   ├── metrics.py       # 进程内指标收集（可观测最小实现）
│   ├── llm.py           # LLMProvider 抽象 + Mock + OpenAICompatible
│   ├── embedder.py      # Embedder 抽象 + Mock(哈希) + Real
│   ├── retriever.py     # 双路混合检索：倒排(BM25-lite) + 向量 + RRF 融合
│   ├── wiki.py          # 聚合层（raw→wiki：实体/概念聚合页，确定性规则版）
│   ├── rag.py           # 管线编排：聚合层优先 + 混合检索 + LLM 生成
│   ├── eval.py          # golden-set 评测（召回命中率 + 忠实度）
│   └── static/index.html # 内部体验入口（黑金主题对话 UI）
├── data/
│   ├── corpus/          # 知识库（维度映射品牌真实经营切面，14 篇含竞品玛芝莲）
│   ├── wiki/            # 聚合索引（entity/concept 聚合页，启动自动生成）
│   ├── business_facts.json / store_facts.json  # 数据真相源
│   └── eval_golden.json # 评测黄金集
├── docs/
│   └── business-case.md # 业务方案、简历 bullet、叙事框架
├── tests/test_main.py   # pytest（全程 Mock，离线可跑）
├── seed_corpus.py       # 灌库脚本
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml / .dockerignore / .env.example
└── README.md
```

---

## 五、业务价值（痛点 → 架构 → 收益）

### 整合系统视角：不是单个工具，是一套从数据到门店日常的系统

1. **数据整合**：打通 ERP 进销存、门店收银、公域（抖音/美团）、私域 SCRM 四类数据源，统一口径（同一个数一个值），建成 47 家门店一条数据链路。
2. **AI 整合**：AI 在统一数据底座上产出六大类应用——知识库、经营分析、内容生产、竞品拆解、行业研究、看板自动化。
3. **嵌入日常**：AI 嵌入门店日常分析工作，员工自然语言查标准/查活动/查供应链；店长为验收方，使用统计 + 零使用预警闭环。
4. **效果量化**：经营分析 66h → 4h、内容生产 21h → 3h、行业研究 2 周 → 1.5 天；年度节约约 5,800 工时（≈3.2 人力年）。

### 痛点 → 架构 → 收益对照

| 痛点 | 架构杠杆 | 预期收益（方案目标） |
|---|---|---|
| 一线查 SOP 慢、易走样 | 向量召回 + LLM 生成 + 原文引用 | 查询从分钟级→秒级，答案可溯源 |
| 竞品情报难检索 | 竞品维度结构化入库、可追问 | 竞品咨询人工承接量目标 ↓30% |
| 知识更新滞后 | ingest 接口随业务即入；应用启动自动比对 corpus 哈希，源文件改动即重建索引（自愈合，防「改源忘灌库」） | 知识同步从月度→实时 |
| 加盟政策咨询重复 | 政策问答 + 出处引用 | 降低人工客服成本 |
| 分析资产难沉淀 | QA 落库 + 评测闭环 | 知识资产化，支撑复盘决策 |
| 多系统数据割裂 | 统一数据口径 + 一条数据链路 | 同一个数一个值，分析可信 |
| AI 用了没人用 | 店长验收 + 使用统计 + 零使用预警 | 47 家门店全部正向、无一家倒退 |

---

## 六、接口一览

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| GET | `/health` | 健康检查（返回当前 llm/embedder 模式） | ✅ |
| POST | `/ingest` | 灌入一篇文档（自动切分 + 向量化） | ✅ |
| POST | `/chat` | RAG 问答，返回 answer + citations + latency | ✅ |
| GET | `/metrics` | 请求数 / 对话数 / 平均延迟 / token | ✅ |
| POST | `/eval` | 基于 golden set 跑评测，返回报告 | ✅ |

> 仅 `GET /`、`/docs`、`/openapi.json`、`/redoc` 免鉴权（便于内部体验）；业务接口仍强制 `X-API-Key`。

---

## 七、评测闭环（AI 质量不是玄学）

```bash
python -m app.eval
```

输出示例（Mock 模式）：
```json
{
  "total": 8,
  "retrieval_hit_rate": 1.0,
  "faithfulness_rate": 1.0,
  "note": "离线 MockProvider：召回命中率验证检索与切分正确；faithfulness 仅在接入真实 LLM 后具备真实意义。"
}
```

- **retrieval_hit_rate**：期望文档是否进入召回 top-3 —— 验证切分 / 向量化 / 检索正确性。
- **faithfulness_rate**：期望事实是否出现在「回答」或「召回原文」中 —— 检索充分性度量。
  > 接入真实 LLM 后，应使用 **LLM-Judge / NLI** 度量生成答案本身的忠实度（ADR-002）。

`pytest` 同样全程 Mock 跑通（7 个用例：鉴权 401 / 422 / ingest+chat / metrics / eval / health）。

完整评测报告（逐题明细 + Badcase 分析 + 成本/延迟估算）见 [docs/eval-report.md](docs/eval-report.md)。

### 评测门禁（LLMOps CI，2026-10-30 新增 / 2026-11-03 加固）

```bash
python -m app.eval_cli --mock            # 跑 8 用例，默认阈值 0.95（与 CI 一致），生成 Markdown 报告 + 写 history
python -m app.eval_cli --mock --threshold 0.9    # 自定义阈值（放宽时请说明理由）
python -m app.eval_cli --mock --json      # 输出 JSON 供 CI 解析
```

- 召回命中率低于阈值时**非零退出码（2）**——CI 可据此阻断合并（失败拦截）
- `--mock` 强制离线 MockProvider + rule 编译，**不受上游 LLM 限流影响，CI 可复现**
- 评测前**强制重建索引**（清空 document/chunk 后从 `data/corpus` 重灌）：保证 CI 空库可跑，
  同时清除本地 `/ingest` 留下的测试脏数据。副作用是本操作会丢弃手工灌入的文档，评测只认 corpus 真相源
- 产物：`reports/eval_report_<时间戳>.md` + `reports/history.json`（监控看板数据源），CI 中以 artifact 上传
- CI 顺序：`pytest`（含检索契约测试）→ 评测门禁，任一失败即红

### 语义探针（非门禁，2026-11-03 新增）

```bash
python -m app.eval_probe                  # mock vs local 对照
python -m app.eval_probe --embedders mock --json
```

- 目的：单独量化「向量层的语义泛化」，与召回门禁解耦。golden 门禁测「整套检索对不对」，
  探针测「换掉向量实现差多少」——两者混在一起会互相掩盖。
- **退出码恒为 0**：CI 无本地模型，local 降级为 mock，若接门禁会每次失败。
  因此 CI 里只做 `continue-on-error` 冒烟，真正的对照在装有模型的机器上跑。
- 产物：`reports/probe_<时间戳>.md`（逐题对照，标注 ✅ 命中 / ⚠️ 仅 top3 / ❌ 未召回）


### 监控看板（LLMOps 可观测，2026-10-30 新增）

启动服务后访问 `/monitor`（免鉴权，黑金主题）：

- 实时 KPI：总请求 / 问答请求 / 平均延迟 / Token 消耗（每 5 秒刷新，读 `/metrics`）
- 评测历史：读 `reports/history.json` 展示召回率/忠实度趋势与通过状态

LLMOps 设计文档（含与生产级差距的诚实边界）见 [docs/llmops.md](docs/llmops.md)。

**技术博客**：
- <a href="docs/blog/rag-data-trust.md" target="_blank">《从"答错 43 家门店"到数据可信：一个 RAG 知识库的可靠性实战》</a>——完整复盘数据事故、单一真相源、自愈索引、接地测试与评测闭环的落地过程。
- <a href="docs/blog/rag-eval-reliability.md" target="_blank">《从 83% 到 100%：一个 RAG 知识库评测体系的可靠性实战》</a>——评测通过率从 83.3% 到 24/24（100%）的完整过程：口语化检索根治、评测词表边界修正、权威闸收敛共享模块、知识治理 80→0 分级闭环。
- <a href="docs/blog/llmops-production-monitoring.md" target="_blank">《从离线门禁到生产监控：RAG 知识库的 LLMOps 补全实战》</a>——零依赖 Langfuse 生产流量采样监控、RAGAS 四指标量化，以及评测回归暴露并修复聚合页口径漂移的实战。
- <a href="docs/blog/semantic-cache-query-rewrite.md" target="_blank">《语义缓存与查询改写：RAG 生产级化的最后两块拼图》</a>——零依赖 SQLite 语义缓存（余弦命中、省成本降延迟）+ LLM 查询改写（质量校验、静默降级），25 项单元测试与端到端验证。
- <a href="docs/blog/feedback-ci-dashboard.md" target="_blank">《从反馈闭环到 CI 门禁：RAG 知识库的生产运维补全》</a>——用户点赞/点踩沉淀 golden set、GitHub Actions 评测门禁（mock 离线可复现）、零依赖可视化运维看板，13 项单元测试与端到端验证。
- <a href="docs/blog/zero-dep-doc-parsing.md" target="_blank">《零依赖文档深度解析：让 docx、压缩 PDF 与表格进入 RAG》</a>——纯标准库解析 docx 段落+表格、PDF FlateDecode 压缩流、xlsx 表格转 Markdown，修复 4 个真实 bug，9 项单元测试与真实文件验证。
- <a href="docs/blog/ai-adoption-methodology.md" target="_blank">《AI 落地五步法：从 47 家门店的实战中提炼的跨行业方法论》</a>——把真实 AI 落地经验抽象为「诊断→数据→闭环→评测→规模」五步法，附零售/制造/医疗/教育跨行业推演。

---

## 八、真实 LLM 链路验证记录（2026-10-22）

> 证明：本服务的 `OpenAICompatibleProvider` 不止能跑 Mock，也能接真实 LLM 端到端产出可用答案。

**验证环境**：沙箱直连商汤真实上游 `https://token.sensenova.cn/v1`（走代理可达；注：`git` 隧道被沙箱策略拦 502，但 LLM REST 端点正常）。

**选用模型**：`sensenova-6.8-flash-lite`（商汤网关 `keys.json` 中的可用模型之一）。

**踩到的坑（已修复）**：该模型是**推理模型**，默认把 token 预算全花在 `reasoning` 思考链上，`message` 里没有 `content` 字段，原 `data["choices"][0]["message"]["content"]` 会取空。
修复方式（已合入 `app/llm.py` + `app/config.py`）：请求体带 `"reasoning":{"enabled":false}`，并做 `content → reasoning_content` 兜底提取；开关 `RAG_LLM_DISABLE_REASONING`（默认 `true`）。

**`.env` 配置（真实链路）**：
```ini
RAG_LLM_PROVIDER=real
RAG_LLM_BASE_URL=https://token.sensenova.cn/v1
RAG_LLM_API_KEY=sk-xxxx          # 取自本地商汤网关 keys.json（acc1/acc2/acc3 三把可轮询）
RAG_LLM_MODEL=sensenova-6.8-flash-lite
RAG_LLM_DISABLE_REASONING=true   # 关闭思考链，直出 content
# 向量化保持 mock（与已存库 dim=256 一致）
```

**验证结果（mock 向量灌库 13 篇 → 真实 /chat，真实网络往返 3.2–4.7s）**：

| 问题 | 真实 LLM 回答 | 召回来源 |
|---|---|---|
| 天鹅公主是哪个城市的品牌？ | **清远**（无 RAG 时模型幻觉为「上海」，RAG 检索上下文纠正为清远并引《品牌介绍》原文） | 《品牌介绍：天鹅公主 SWAN NOBLE》 |
| 2026 夏季推出了哪几款新品？ | 黑金可颂 / 联名桃酥 / 荔枝玫瑰冻芝士 / 青提冷萃欧包 | 《2026 夏季四款新品》 |
| 目前有多少家门店？ | 47 家（清远 40 / 广州 4 / 肇庆 2 / 佛山 1） | 《门店网络》 |

> 关键证据：第 1 问证明 **RAG 检索上下文成功压制了模型参数幻觉**——正是 ADR-002「没有评测闭环的系统不可上线」的落地演示。

**说明**：真实 Key 仅存于本地 `.env`（已被 `.gitignore` 排除，**未进入本仓库**）；远程仓库不含任何 Key。代码改进见 commit `52fe027`。

---

## 九、落地状态（诚实标注）

| 模块 | 状态 |
|---|---|
| 品牌知识维度建模（默认 14 篇语料映射真实经营切面） | ✅ 已建模（默认语料为脱敏/合成映射，生产可替换） |
| RAG 架构实现（FastAPI 全家桶 + 抽象解耦） | ✅ 已实现 |
| 真实 LLM 端到端验证 | ✅ 已验证（sensenova-6.8-flash-lite） |
| 评测闭环（golden-set） | ✅ 已建立 |
| 容器化部署（Docker / compose） | ✅ 已就绪，可一键起 |
| 门店智能体 POC 方案与指标体系（巡检准确率/店长减负率/单店 ROI） | 🟡 方案已定义，指标为试点目标值（详见 [docs/business-case.md](docs/business-case.md) 第五节） |
| **接入本地生产知识库** | ✅ **支持**：设置 `RAG_CORPUS_DIR` 指向真实语料目录（如 `D:\WorkBuddy知识库\wiki`），递归扫描 entity/concept/comparison/synthesis 全部 md，重建索引即可对真实知识问答 |
| 生产级向量库（pgvector）/ 多租户 | 🟡 规划中 |

> 本仓库默认语料为**脱敏/合成展示**（可公开）。**生产使用**时，将 `RAG_CORPUS_DIR` 指向本地真实知识库（见上文），即可对真实品牌/竞品/营销知识问答；真实数据不推送到公开仓库。

---

## 十、Docker（内部体验 / 部署就绪）

```bash
cp .env.example .env        # 按需填真实 Key
docker compose up --build
# 默认 8000 端口；容器启动自动比对 corpus 哈希并重建索引（源文件改动即自愈合）
```

### Kubernetes 部署（2026-10-30 新增）

生产规范 K8s 清单已就绪（Deployment / Service / HPA / Ingress / Secret / Kustomize）：

```bash
# 构建镜像
docker build -t swan-rag:latest .

# 配置 Secret（填入真实 Key）
cd k8s && cp secret.example.yaml secret.yaml

# 一键部署
kubectl apply -k k8s/

# 验证
kubectl get pods -l app=swan-rag
kubectl get hpa swan-rag
```

- **Deployment**：2 副本，资源限制，readiness/liveness 健康检查
- **HPA**：CPU 超 70% 自动扩容，最多 6 副本
- **Secret**：API Key / LLM Key 从 Secret 注入，不写进镜像
- 完整手册（含扩缩容验证、FAQ、生产补充）见 [docs/k8s-deploy.md](docs/k8s-deploy.md)

> ⚠️ 诚实标注：清单已按生产规范编写并通过 YAML 语法校验，但**未在真实集群执行**（本机无 Docker 守护进程，无法启动本地 K8s）。在有 Docker / 云 K8s 环境按手册即可部署。

### 数据护栏（防错误资料回流）

知识库曾因 corpus 凭空编造「43 家含韶关」且改源后未重建索引，长期返回错误门店数。现已设多道护栏：

1. **唯一真相源（Single Source of Truth）**：门店数据只写在 `data/store_facts.json`（47 家：清远含连州 40 / 广州含花都 4 / 肇庆 2 / 佛山 1，韶关不存在）；门店以外的关键业务事实收在 `data/business_facts.json`（新品上市日期、加盟门槛、包材规格等）。两者都是**唯一更新入口**。
2. **源—索引一致（自愈合）**：`app/seed.py` 在启动时比对 `data/corpus/*.md` 内容哈希，库空或源漂移即清旧建新，杜绝「改源忘灌库」。
3. **内容接地校验（CI 卡点）**：
   - `tests/test_grounding.py`：门店数据**数据无关一致性**校验——真相源自洽、corpus 所有「X 家门店」表述须等于真相源总数、分布须吻合、不得出现真相源之外的城市。
   - `tests/test_business_grounding.py`：业务事实**必现校验**——`business_facts.json` 中每条事实的关键词必须出现在 corpus（新品/加盟/包材/会员/供应链/运维/数字化）。
   - **pytest 不通过即阻断合并**。

#### 门店数据更新流程（新数据进来时）

护栏不写死任何数字，因此真实数据更新不会被误伤，只需按以下顺序操作：

```bash
# 1) 改唯一真相源（例如新增门店、总数变化、新增城市）
#    编辑 data/store_facts.json
# 2) 由真相源重建门店网络文件（不再手写，杜绝漂移）
python scripts/generate_stores_md.py
# 3) 同步其余 corpus 叙述（如 02_夏季新品 的「全国 X 家门店同步上市」）使其吻合
# 4) 跑测试确认接地一致
pytest tests/test_grounding.py -q
# 5) 启动应用，自愈合自动重建向量索引
uvicorn app.main:app --reload
```

#### 业务事实更新流程（门店以外，如新品/加盟/包材）

```bash
# 1) 改唯一真相源：编辑 data/business_facts.json（改 value / 改 must_appear）
# 2) 同步 data/corpus/*.md 中的对应叙述
# 3) 跑测试确认必现关键词全部命中
pytest tests/test_business_grounding.py -q
# 4) 启动应用，自愈合自动重建向量索引
uvicorn app.main:app --reload
```

任一步遗漏（如某文件忘了随真相源更新），对应测试会直接失败并指出缺失关键词，阻断错误资料上线。

若任一步遗漏（如某文件忘了随真相源更新），第 4 步测试会直接失败并指出矛盾门店数，阻断错误资料上线。

---

## 十一、LoRA 领域微调实验（2026-10-30 新增）

在品牌知识语料上做 Qwen2.5-0.5B 的 LoRA 领域微调，让模型回答更贴合知识库口径（与 RAG 互补）。

```bash
# 数据准备（corpus → 指令数据集）
python finetune/prepare_data.py

# 训练（需 GPU 环境）
python finetune/train_lora.py --model Qwen/Qwen2.5-0.5B --epochs 3

# 微调前后效果对比
python finetune/eval_lora.py --base-model Qwen/Qwen2.5-0.5B --adapter finetune/output/lora_adapter
```

- 数据流水线：`data/corpus/*.md → finetune/data/train.jsonl → LoRA 适配器 → 对比报告`
- 训练配置：LoRA r=8 / alpha=16 / lr=2e-4 / 3 epochs / 0.5B 基础模型
- 实验设计与诚实标注见 [finetune/README.md](finetune/README.md)

> ⚠️ 诚实标注：脚本可复现、数据已生成，但**本机无 GPU 未实际训练**；在有 GPU 环境按 README 即可运行。

---

## 十二、性能压测（2026-10-30 新增）

对 `/chat` 接口做并发压测（`perf/load_test.py`），并完成针对性优化：

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| **QPS** | 56.69 | **182.27** | **+222%** |
| **P50 延迟** | 1130 ms | **469 ms** | **-59%** |
| **P95 延迟** | 2225 ms | **555 ms** | **-75%** |
| **P99 延迟** | 3667 ms | **611 ms** | **-83%** |
| 错误率 | 0.00% | 0.00% | — |

优化措施（全部零外部依赖）：

1. **SQLite WAL + 并发 PRAGMA**（`app/database.py`）：读写不互斥，消除并发写串行 + fsync
2. **聚合索引落盘节流**（`app/wiki.py`）：仅构建时写盘，避免每次请求并发写同一文件
3. **QA 审计异步批量落库**（`app/rag.py`）：请求只入内存队列，后台线程批量写，不再等待 commit
4. **检索数据进程级缓存**（`app/retriever.py`）：chunk / 倒排索引 / 文档标题常驻内存，
   指纹 = corpus 哈希 + 数据库状态（单条 SQL），`/ingest` 后自动失效（已修复并验证）

复现命令：

```bash
RAG_LLM_PROVIDER=mock RAG_WIKI_COMPILER=rule python -m uvicorn app.main:app --port 8000
python perf/load_test.py --url http://127.0.0.1:8000/chat --concurrency 100 --requests 300
```

完整瓶颈分析、逐步验证数据、扩展性对照与经验总结见 [docs/perf-report.md](docs/perf-report.md)。

---

## 十三、架构决策记录（ADR）

**ADR-001 · 供应商解耦（LLM / Embedder 抽象）**
> 把 LLM 与向量化抽象成接口，内置 `Mock` + `OpenAICompatible` 两套实现。
> 动机：① 离线 / CICD 可跑通，不被外部 API 与网络绑架；② 真实供应商可插拔，
> 避免被单一厂商锁定；③ 演示「架构师该做的抽象」，而非「调通一个接口」。

**ADR-002 · 评测闭环优先**
> 必须有 golden-set 评测，把「AI 质量」变成可量化指标。离线用 Mock 验证检索正确性；
> 真实环境用 LLM-Judge 度量生成忠实度。动机：AI 应用最大的坑是「看着能答，实则幻觉」，
> 没有评测闭环的系统不可上线。

**ADR-003 · 复用已学后端能力，而非另起炉灶**
> 直接复用 FastAPI + SQLModel + Depends + 统一错误信封 + 配置分离 + 全局鉴权 + Docker
> （见同系列 `store_tool`）。动机：展示「能力递进」——同一人会从 CRUD 进阶到
> AI 架构，而不是两段不相关的代码。

**ADR-004 · 可靠性设计文档化**
> 失败模式（FMEA）、兜底路径、可恢复性与可观测性设计单独成文：
> [docs/reliability.md](docs/reliability.md)。动机：可靠性不是「代码里顺手加的防御」，
> 而是可评审、可演进、可复盘的设计产物。

**ADR-005 · raw → wiki → RAG 双轨（聚合层优先）**
> 检索不直接只走原始 chunk，而是先查聚合层（`app/wiki.py`：实体/概念聚合页，
> 含摘要、关键事实、来源列表）。命中聚合页则优先使用聚合上下文，再补充原始 chunk 细节。
> 动机：把 llm-wiki 的「LLM 参与知识组织」思路落地为可运行实现——同一实体多源聚合为一页，
> 答案可溯源到来源列表。
> 实现：`RAG_WIKI_COMPILER=llm` 时调用真实 LLM 识别实体/概念并生成聚合页（已验证
> sensenova-6.8-flash-lite 一次编译产出 15 个聚合页，跨源聚合生效）；
> `=rule` 时用确定性词表聚合（离线可测）。LLM 输出解析失败自动降级规则兜底。

**ADR-006 · 混合检索（倒排 BM25 + 向量 + RRF 融合）**
> 召回层双路并行：向量路处理语义（口语化/换说法），倒排路处理精确术语（业务词/编号），
> 用 RRF（Reciprocal Rank Fusion）融合取 top-k。
> 动机：单路向量在垂直术语场景可能漏精确命中；单路倒排在口语化场景漏语义。
> 双路取长补短——这是 llm-wiki「词频饱和 + 业务子库加权」生产检索的通用化实现。
> 注：当前环境无真实 embedding 服务（商汤网关不支持 /embeddings），向量路使用 mock；
> 接入真实 Embedder 后（`get_embedder()` 已支持 real），口语化语义召回能力即生效。

---

## 十四、与 `store_tool` 的关系

`store_tool` 是「后端最小项目」（纯 CRUD + 鉴权 + 迁移 + Docker）。
本仓库在**同一套后端能力之上加了一层 AI 应用架构**，构成「从 CRUD 到 AI 架构」的能力证据链。

- 仓库：https://github.com/Babymrbbbb/store_tool
- 定位：store_tool 是后端底座，本仓库在其之上叠加 AI 应用架构，二者构成「从 CRUD 到 AI 架构」的能力证据链。

---

## 许可

教学 / 作品集用途。知识库内容为合成数据，与任何真实企业经营数据无关。
