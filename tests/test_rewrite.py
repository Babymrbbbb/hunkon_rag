"""查询改写测试：开关、可改写判断、质量校验、缓存、降级。

背景：
  生产知识库 query_rewrite.py（2026-11-03 P0-4），hunkon_rag 移植为 app/rewrite.py。
  核心：口语化问法经 LLM 改写提升召回；失败静默降级；默认关闭。
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from app.rewrite import (  # noqa: E402
    QueryRewriteCache,
    _keeps_core_terms,
    _looks_rewritable,
    rewrite_query,
)


def test_looks_rewritable():
    assert _looks_rewritable("那家卖坚果礼盒的怎么样？")  # 指代词
    assert _looks_rewritable("Hunkon 一共有多少家门店？")  # 口语词「多少家」
    assert _looks_rewritable("他们最近在干嘛？")  # 指代词 + 短
    assert not _looks_rewritable("Hunkon 门店网络")  # 规范
    assert not _looks_rewritable("玛芝莲超级工厂战略")  # 规范


def test_keeps_core_terms():
    assert _keeps_core_terms("Hunkon 有多少家门店？", "Hunkon 共有多少家门店？")
    assert _keeps_core_terms("玛芝莲最近在干嘛？", "玛芝莲最近在做什么？")
    assert _keeps_core_terms("门店数是多少？", "Hunkon 门店数是多少？")  # 门店 2 字词
    assert not _keeps_core_terms("Hunkon 有多少家门店？", "请提供更具体的信息")  # 丢关键实词


def test_cache_put_get():
    db = Path(tempfile.mkdtemp()) / "rewrite.db"
    c = QueryRewriteCache(db_path=str(db))
    c.put("问题", "改写后")
    assert c.get("问题") == "改写后"
    assert c.get("不存在") is None
    c.close()


def test_rewrite_disabled_returns_original():
    os.environ["RAG_QUERY_REWRITE"] = "0"
    q, changed = rewrite_query("那家卖坚果礼盒的怎么样？")
    assert q == "那家卖坚果礼盒的怎么样？"
    assert changed is False


def test_rewrite_enabled_mock_degraded(monkeypatch):
    """开启改写但 LLM 是 mock（输出含回显原文）时，应通过质量校验并返回改写。"""
    os.environ["RAG_QUERY_REWRITE"] = "1"

    class FakeLLM:
        def complete(self, system, user):
            return user  # 回显原文

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: FakeLLM())
    q, changed = rewrite_query("那家卖坚果礼盒的怎么样？")
    # 回显原文 → 与原文相同 → 视为无需改写，降级原样
    assert q == "那家卖坚果礼盒的怎么样？"
    assert changed is False
    os.environ["RAG_QUERY_REWRITE"] = "0"


def test_rewrite_enabled_llm_error_degraded(monkeypatch):
    """LLM 抛异常时应静默降级为原文，不阻塞。"""
    os.environ["RAG_QUERY_REWRITE"] = "1"

    class BoomLLM:
        def complete(self, system, user):
            raise RuntimeError("429")

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: BoomLLM())
    q, changed = rewrite_query("那家卖坚果礼盒的怎么样？")
    assert q == "那家卖坚果礼盒的怎么样？"
    assert changed is False
    os.environ["RAG_QUERY_REWRITE"] = "0"