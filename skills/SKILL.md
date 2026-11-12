---
name: rag-ruling
description: 检索 / 聚合链路的三道确定性护栏技能包（口径覆盖、同义扩展相关性过滤、空输入失败），可独立 load 或 compose 成管线。
type: skill-package
version: 1.0.0
modes: [A, B, C]
---

# rag-ruling 技能包

把「确定性护栏三模式」封装成可被 AI 编码 Agent 按需调用的技能集合。
灵感来自 `addyosmani/agent-skills`（给 AI 编码 Agent 用的生产级工程技能包）——
区别在于本包聚焦**检索 / 聚合链路的正确性护栏**，且全部为合成演示的方法论，
不依赖任何真实业务数据。

## 包含的技能

| 技能名 | 模式 | 解决什么 | 入口函数 |
|---|---|---|---|
| `rag-ruling/ruling-override` | A | 口径覆盖必须落在**每处**执行路径 | `apply_ruling_overrides` / `register_ruling_path` / `verify_ruling_coverage` |
| `rag-ruling/query-rewrite` | C | 相关性过滤先做**同义扩展**，避免同义表述被字面匹配误杀 | `expand_query` / `relevance_filter` |
| `rag-ruling/source-guard` | B | **空输入比报错更危险**：聚合入口有效源为空必须显式失败 | `require_nonempty_sources` |

## 设计原则（与工程铁律的对应）

- **模式 A**：凡「LLM 生成后确定性覆盖」的闸，必须在所有实际输出路径上 import 复用同一份函数——只改一处 = 形同虚设。`register_ruling_path()` + `verify_ruling_coverage()` 在 CI 中证明闸落在了每处路径。
- **模式 B**：拿 0 个源去问 LLM 会产出空壳页且能过字段校验，覆盖掉完整旧版。故必须显式失败，而不是静默继续。
- **模式 C**：字面匹配会误杀相关文档（如「复购提升」字面搜不到「回购」）。过滤前必须先做同义扩展。

## 用法

```python
from skills import load_skill, run_guard_pipeline

# 单技能加载
spec = load_skill("rag-ruling/source-guard")
print(spec["entrypoints"])  # ['require_nonempty_sources']

# 组合管线（三道闸一次性串起来）
out = run_guard_pipeline(
    query="会员复购提升",
    docs=[{"text": "会员回购率环比上升"}],
    generated_text="演示品牌共 40 家门店",
    page_name="演示品牌",
)
```

## 验证

`pytest tests/test_skill_package.py` 覆盖：技能注册完整性、三入口函数可调用、
`run_guard_pipeline` 端到端（空源抛错 / 同义召回 / 口径覆盖生效）、
`verify_ruling_coverage` 在登记后归零。
