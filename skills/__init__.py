"""rag-ruling 技能包：把「确定性护栏三模式」封装为可组合、可独立调用的 Agent 技能。

设计动机（对应 addyosmani/agent-skills 范式）：
  内部工程沉淀出一套「检索 / 聚合链路正确性」护栏（源于知识库运维铁律的通用化），
  原本以散落的函数形式存在于 app/rules.py。把它们显式封装成「技能包」后，
  AI 编码 Agent 可以按需 load 单个技能、或 compose 成最小管线，无需理解实现细节，
  也能保证「口径覆盖 / 相关性过滤 / 空输入失败」三道闸不被漏掉。

本包不依赖任何真实业务数据，全部为合成演示的方法论。三技能映射：
  - rag-ruling/ruling-override  → 模式 A：口径覆盖必须落在每处执行路径
  - rag-ruling/query-rewrite    → 模式 C：相关性过滤先做同义扩展
  - rag-ruling/source-guard     → 模式 B：空输入比报错更危险（显式失败）

实现全部复用 app/rules.py，本文件只做「技能注册 + 组合管线」的薄封装。
"""
from __future__ import annotations

from app.rules import (
    RulingError,
    apply_ruling_overrides,
    expand_query,
    register_ruling_path,
    relevance_filter,
    require_nonempty_sources,
    verify_ruling_coverage,
)

SKILLS: dict[str, dict] = {
    "rag-ruling/ruling-override": {
        "title": "权威口径确定性覆盖",
        "mode": "A",
        "entrypoints": [
            "apply_ruling_overrides",
            "register_ruling_path",
            "verify_ruling_coverage",
        ],
        "trigger": "任何 LLM 生成后要对齐权威口径、且需保证闸落在每处输出路径时",
    },
    "rag-ruling/query-rewrite": {
        "title": "同义扩展 + 相关性过滤",
        "mode": "C",
        "entrypoints": ["expand_query", "relevance_filter"],
        "trigger": "检索 / 聚合前做相关性过滤，避免同义表述被字面匹配误杀时",
    },
    "rag-ruling/source-guard": {
        "title": "空输入显式失败",
        "mode": "B",
        "entrypoints": ["require_nonempty_sources"],
        "trigger": "聚合 / 生成入口收到源列表，必须以「有效源 ≥ 1」前置校验时",
    },
}


def load_skill(name: str) -> dict:
    """按名称加载一个技能的定义（标题 / 模式 / 入口函数 / 触发条件）。

    这是 Agent 调用技能包的标准入口：先 load 拿到定义，再调用对应 entrypoint。
    """
    if name not in SKILLS:
        raise KeyError(f"未知技能: {name}（可用: {sorted(SKILLS)}）")
    return dict(SKILLS[name])


def list_skills() -> list[str]:
    return sorted(SKILLS)


def run_guard_pipeline(
    *, query: str, docs: list, generated_text: str, page_name: str = ""
) -> dict:
    """组合三技能的最小可演示管线（纯方法论，合成）。

    顺序即三道闸的天然依赖：
      1) source-guard  → 先确认有效源非空（空源直接抛 RulingError）
      2) query-rewrite → 再对源做同义扩展后的相关性过滤
      3) ruling-override→ 最后对生成文本做确定性口径覆盖
    """
    require_nonempty_sources(docs)  # 技能 B
    kept = relevance_filter(docs, query)  # 技能 C
    guarded = apply_ruling_overrides(generated_text, page_name=page_name)  # 技能 A
    return {"kept_docs": kept, "guarded_text": guarded}


__all__ = [
    "SKILLS",
    "load_skill",
    "list_skills",
    "run_guard_pipeline",
    "RulingError",
]
