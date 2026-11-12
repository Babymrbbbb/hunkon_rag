---
name: rag-ruling/query-rewrite
description: 检索 / 聚合前先做同义扩展，再按扩展 token 做相关性过滤，避免同义表述被字面匹配误杀（模式 C）。
mode: C
entrypoints: [expand_query, relevance_filter]
---

# rag-ruling/query-rewrite（模式 C）

## 触发条件
在检索或聚合前需要做相关性过滤，且你担心**字面匹配会漏掉同义表述**。

## 为什么需要它
内部实战里，一条「复购提升」的查询因字面匹配搜不到文档里写的「回购」，
相关性召回从 25 掉到 2，直接误杀了大量相关文档。根因：过滤前没有做同义扩展。

## 怎么用
1. `tokens = expand_query(query)` —— 把查询里的词按同义词表扩展成 token 集合。
2. `kept = relevance_filter(docs, query, required=1)` —— 按扩展后的 token 命中计数过滤，
   命中数 ≥ required 的文档保留。

## 反模式
- ❌ 直接用 `query in doc` 做字面包含判断。
- ❌ 同义词表只在检索端用、过滤端不用（两端必须一致）。
- ❌ required 设得过高导致正常相关文档被过滤。
