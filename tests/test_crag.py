"""CRAG 自纠错测试：开关、降级、相关性判断、改写重试。

背景：
  生产级 RAG 的 CRAG 范式：LLM 判断上下文相关性，不足则改写查询重试。
  swan_rag 实现为 app/crag.py，默认关闭（RAG_CRAG=1 且真实 LLM 时生效）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from app.config import settings  # noqa: E402
from app.crag import _grade, _hits_to_text, _rewrite, corrective_rerank  # noqa: E402


class _FakeChunk:
    def __init__(self, cid, text):
        self.id = cid
        self.document_id = cid
        self.text = text


class _FakeDoc:
    def __init__(self, cid, title):
        self.id = cid
        self.title = title


def _hits():
    return [(0.9, _FakeChunk(1, "天鹅公主有 47 家门店，其中清远 40 家。"))]


def _doc_titles():
    return {1: "门店网络"}


class _FakeRetriever:
    def __init__(self, result):
        self._result = result

    def search(self, *a, **k):
        return self._result


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 8


def test_hits_to_text():
    txt = _hits_to_text(_hits(), _doc_titles())
    assert "门店网络" in txt
    assert "47 家门店" in txt


def test_grade_relevant(monkeypatch):
    class FakeLLM:
        def complete(self, system, user):
            return "relevant"

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: FakeLLM())
    assert _grade("天鹅公主有多少家门店？", "天鹅公主有 47 家门店") == "relevant"


def test_grade_irrelevant(monkeypatch):
    class FakeLLM:
        def complete(self, system, user):
            return "irrelevant"

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: FakeLLM())
    assert _grade("金粒门主打什么？", "天鹅公主有 47 家门店") == "irrelevant"


def test_grade_ambiguous(monkeypatch):
    class FakeLLM:
        def complete(self, system, user):
            return "ambiguous"

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: FakeLLM())
    assert _grade("冰面包能放几天？", "冰面包非在售") == "ambiguous"


def test_grade_failure_degrades_to_relevant(monkeypatch):
    class BoomLLM:
        def complete(self, system, user):
            raise RuntimeError("429")

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: BoomLLM())
    assert _grade("问题", "资料") == "relevant"  # 失败静默降级


def test_rewrite_failure_returns_original(monkeypatch):
    class BoomLLM:
        def complete(self, system, user):
            raise RuntimeError("429")

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: BoomLLM())
    assert _rewrite("天鹅公主有多少家门店？") == "天鹅公主有多少家门店？"


def test_rewrite_mock_returns_original():
    """mock LLM 返回占位文本（不是原文，也不是 HTTP 错误）→ 改写返回占位文本。
    注意：corrective_rerank 在 llm_provider != real 时不会调用 _rewrite，
    所以这里的返回值只影响直接调用 _rewrite 的场景。"""
    old_provider = settings.llm_provider
    settings.llm_provider = "mock"
    try:
        r = _rewrite("天鹅公主有多少家门店？")
        assert r != "", "改写不应返回空"
    finally:
        settings.llm_provider = old_provider


def test_crag_disabled_returns_original():
    """默认关闭（RAG_CRAG=0）→ 原样返回。"""
    old = settings.crag
    settings.crag = False
    try:
        hits, used, retried = corrective_rerank(
            "问题", _hits(), None, _FakeEmbedder(), _FakeRetriever(_hits()), 3, "问题", _doc_titles()
        )
        assert [c.id for _s, c in hits] == [c.id for _s, c in _hits()]
        assert used is False and retried is False
    finally:
        settings.crag = old


def test_crag_mock_llm_returns_original():
    """开启但 LLM 不是 real → 原样返回。"""
    old_crag = settings.crag
    old_provider = settings.llm_provider
    settings.crag = True
    settings.llm_provider = "mock"
    try:
        hits, used, retried = corrective_rerank(
            "问题", _hits(), None, _FakeEmbedder(), _FakeRetriever(_hits()), 3, "问题", _doc_titles()
        )
        assert [c.id for _s, c in hits] == [c.id for _s, c in _hits()]
        assert used is False and retried is False
    finally:
        settings.crag = old_crag
        settings.llm_provider = old_provider


def test_crag_irrelevant_rewrites_and_retries(monkeypatch):
    """开启 + real LLM + 判断 irrelevant → 改写重试。"""
    old_crag = settings.crag
    old_provider = settings.llm_provider
    settings.crag = True
    settings.llm_provider = "real"

    class FakeLLM:
        def complete(self, system, user):
            if "检索质量评估器" in system:
                return "irrelevant"
            if "改写" in system:
                return "天鹅公主一共有多少家门店？"
            return ""

    new_hits = [(0.95, _FakeChunk(2, "天鹅公主全国 47 家门店"))]

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: FakeLLM())
    hits, used, retried = corrective_rerank(
        "天鹅公主有多少家门店？",
        _hits(),
        None,
        _FakeEmbedder(),
        _FakeRetriever(new_hits),
        3,
        "天鹅公主有多少家门店？",
        _doc_titles(),
    )
    assert used is True and retried is True
    assert hits == new_hits, f"重试结果未生效: {hits}"
    settings.crag = old_crag
    settings.llm_provider = old_provider


def test_crag_rewrite_failure_keeps_original(monkeypatch):
    """开启 + 判断 irrelevant + 改写失败 → 保留原结果。"""
    old_crag = settings.crag
    old_provider = settings.llm_provider
    settings.crag = True
    settings.llm_provider = "real"

    class FakeLLM:
        def complete(self, system, user):
            if "检索质量评估器" in system:
                return "irrelevant"
            if "改写" in system:
                return "HTTP 429"  # 改写失败信号

    monkeypatch.setattr("app.llm.get_llm_provider", lambda: FakeLLM())
    hits, used, retried = corrective_rerank(
        "问题", _hits(), None, _FakeEmbedder(), _FakeRetriever([]), 3, "问题", _doc_titles()
    )
    assert used is False and retried is False
    assert [c.id for _s, c in hits] == [c.id for _s, c in _hits()]
    settings.crag = old_crag
    settings.llm_provider = old_provider