---
title: 把护栏封装成技能包 —— 从散落函数到 agent-skills 范式
date: 2026-11-12
tags: [rag, guardrail, agent-skills, methodology]
---

# 把护栏封装成技能包

> 方法论笔记（合成演示，不含真实业务数据）。灵感来自 `addyosmani/agent-skills`：
> 给 AI 编码 Agent 用的「生产级工程技能包」。

## 出发点

`app/rules.py` 里沉淀了三道确定性护栏（模式 A/B/C），原本是散落的函数。它们在
代码里能跑、有单测，但有一个问题：**它们还是「实现细节」，不是「可被 Agent 调用的能力」**。

当一个编码 Agent 接手这个项目时，它不会天然知道「生成文本前必须先过口径覆盖闸」。
如果把它显式封装成技能包（`skills/`），Agent 就能：

1. `load_skill("rag-ruling/source-guard")` 拿到定义与触发条件；
2. 在合适的代码位置插入对应的 entrypoint 调用；
3. 用 `run_guard_pipeline` 一次性把三道闸串起来。

## 三道闸映射到技能

| 技能 | 模式 | 入口 |
|---|---|---|
| `rag-ruling/ruling-override` | A 口径覆盖落在每处路径 | `apply_ruling_overrides` / `register_ruling_path` / `verify_ruling_coverage` |
| `rag-ruling/query-rewrite` | C 同义扩展后过滤 | `expand_query` / `relevance_filter` |
| `rag-ruling/source-guard` | B 空输入显式失败 | `require_nonempty_sources` |

## 为什么是「技能包」而不是「再写个函数」

- **可发现**：Agent 不需要读源码就能知道有哪些护栏能力。
- **可组合**：`run_guard_pipeline` 把三道闸的依赖顺序固化下来。
- **可验证**：`verify_ruling_coverage()` 在 CI 证明「闸落在了每处路径」，而不是只改了一处。

## 与内部工程铁律的关系

这三道闸是内部「知识库运维铁律」里「检索 / 聚合链路正确性」三条的**通用化**——
去掉了所有真实业务参数，只保留机制本身，因此可以在开源复刻仓库里安全演示，
而不会泄露任何真实品牌 / 门店 / 经营数据。

## 落地状态

- `skills/__init__.py`：技能注册 + 组合管线。
- `skills/*/SKILL.md`：三个技能的触发条件与反模式。
- `tests/test_skill_package.py`：注册完整性 + 端到端三模式验证。
