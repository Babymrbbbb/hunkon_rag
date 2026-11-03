# 给 44 条自动化 AI 任务加上可观测性：零依赖接入 Langfuse

> 定位：技术类文章（LLMOps、可观测性、模型网关）
> 作者视角：连锁 AI 落地负责人，自研零依赖模型网关
> 配套代码：sensenova-gateway 仓库（`langfuse.js` + 测试 + 接入文档）

---

## 问题：模型调用不可观测，等于盲跑

生产环境跑着 44 条自动化 AI 任务（经营分析、内容生产、行业研究、竞品监控）。如果模型调用不可观测，会面临：

- **单模型故障不可见**：任务静默失败，业务方第二天才发现
- **token 成本不可控**：月底账单超标，不知道哪个任务最贵
- **延迟劣化无感知**：任务超时，影响业务时效
- **失败无法归因**：是限流？是模型？是 prompt？无从定位

我们自研的模型网关是零依赖的（Node.js 单进程，无 npm 包）。接入可观测性时，面临一个选择：**用官方 SDK（引入依赖）还是直接调 HTTP API（保持零依赖）？**

我们选了后者。原因：网关的定位是「轻量、零依赖、可嵌入」，引入 Langfuse SDK 会破坏这个特性。

---

## 方案：零依赖 Langfuse 埋点

Langfuse 提供 HTTP Ingestion API（`POST /api/public/ingestion`），可以直接调用，无需 SDK。核心工作：

### 1. 签名鉴权

Langfuse 使用 `Basic base64(publicKey:timestamp)` 作为 Authorization，并带 `X-Langfuse-Sdk-Signature = HMAC-SHA256(secretKey, timestamp:body)`。用 Node 内置 `crypto` 模块即可实现。

### 2. Trace 事件构造

每次 LLM 调用，构造两个事件：
- `trace-create`：开始一次调用（记录输入、模型名、元数据）
- `generation-create` / `generation-update`：记录生成过程的输入输出、token 用量、状态、延迟

### 3. Usage 提取

OpenAI 兼容接口的响应里带 `usage` 字段。非流式直接从 JSON 提取；流式需要扫描 `data:` 行，从最后一个带 `usage` 的 chunk 提取。

### 4. 埋点位置

在网关的转发层埋点：
- 请求开始：`startTrace`
- 成功返回：`endTrace(status=OK, usage, latency)`
- 失败返回：`endTrace(status=ERROR, errorMessage)`

---

## 关键代码（简化）

```js
// langfuse.js（零依赖）
const crypto = require('crypto');

function buildAuthHeaders(bodyStr) {
  const timestamp = Date.now().toString();
  const basic = Buffer.from(`${publicKey}:${timestamp}`).toString('base64');
  const signature = crypto.createHmac('sha256', secretKey)
    .update(`${timestamp}:${bodyStr}`).digest('hex');
  return {
    'content-type': 'application/json',
    'Authorization': `Basic ${basic}`,
    'X-Langfuse-Sdk-Signature': signature
  };
}

function ingest(events) {
  const body = JSON.stringify({ batch: events });
  // POST {host}/api/public/ingestion，带签名头
}
```

```js
// server.js 中埋点
const lfTrace = langfuse.startTrace({
  name: 'chat-completion',
  input: reqBody,
  metadata: { model, path: pathname }
});
// 成功时
langfuse.endTrace(lfTrace, {
  usage, status: 'OK', latencyMs: Date.now() - started
});
// 失败时
langfuse.endTrace(lfTrace, {
  status: 'ERROR', errorMessage, latencyMs: Date.now() - started
});
```

---

## 验证

写了一个单元测试（6 项断言）验证：
- 非流式/流式 usage 提取正确
- trace 事件构造完整（trace-create + generation-create/update）
- 签名算法可复算（HMAC-SHA256）
- 未配置 key 时安全跳过（不影响网关原功能）

测试覆盖了「禁用时网关行为与原来完全一致」——这是零依赖方案的关键保证。

---

## 效果与经验

接入后可观测性后，每次模型调用都有：
- trace：输入/输出/状态
- generation：模型名、token 用量、延迟、错误
- 可按模型 / key / 任务维度聚合分析

**经验总结**：
1. **零依赖方案完全可行**：Langfuse 的 HTTP API 不复杂，Node 内置模块足够
2. **埋点要在网关层做**：上层任务零改动，只要走网关自动被观测
3. **默认关闭**：不设置环境变量就不上报，保证网关原行为不变
4. **与评测门禁配合**：可观测性（运行时）+ 评测门禁（发布时）+ 执行验收分离（质量），形成完整 LLMOps 闭环

---

> 配套代码见 sensenova-gateway 仓库：`langfuse.js`（零依赖埋点模块）、`test/langfuse.test.js`（单元测试）、`LLMOPS_LANGFUSE.md`（部署与验证指南）。