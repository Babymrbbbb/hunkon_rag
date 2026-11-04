"""精排器测试：ScoreReranker 保持原序、LocalOnnxReranker 模型缺失降级、get_reranker 配置。

背景：
  生产库 rag_qa.py 的 Reranker 体系，hunkon_rag 移植为 app/reranker.py。
  核心：cross-encoder 精排提升检索质量；模型缺失/失败静默降级，绝不伪造能力。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from app.config import settings  # noqa: E402
from app.reranker import (  # noqa: E402
    LocalOnnxReranker,
    Reranker,
    ScoreReranker,
    get_reranker,
)


class _FakeChunk:
    def __init__(self, cid, text):
        self.id = cid
        self.text = text


def _cands():
    return [
        (0.9, _FakeChunk(1, "Hunkon 有 47 家门店")),
        (0.8, _FakeChunk(2, "金粒门主打干货类")),
        (0.7, _FakeChunk(3, "冰面包非在售")),
    ]


def test_abstract_base():
    with pytest.raises(TypeError):
        Reranker()  # noqa: B018 - 抽象类不能实例化


def test_score_reranker_keeps_order():
    rr = ScoreReranker()
    out = rr.rerank("Hunkon 门店", _cands(), top_k=2)
    assert len(out) == 2
    assert out[0][1].id == 1  # 原最高分保持第一
    assert out[1][1].id == 2


def test_score_reranker_respects_top_k():
    rr = ScoreReranker()
    out = rr.rerank("查询", _cands(), top_k=1)
    assert len(out) == 1


def test_local_reranker_degrades_when_model_missing(monkeypatch):
    """模型目录不存在 → 自动降级 ScoreReranker，输出与 score 一致。"""
    # 指向不存在的模型目录
    rr = LocalOnnxReranker(model_dir=ROOT / "_models" / "__not_exist__")
    assert rr.name == "local-cross"
    out = rr.rerank("Hunkon 门店", _cands(), top_k=2)
    # 降级后应保持原序（ScoreReranker 行为）
    assert out[0][1].id == 1
    assert out[1][1].id == 2


def test_local_reranker_empty_candidates():
    rr = LocalOnnxReranker(model_dir=ROOT / "_models" / "__not_exist__")
    assert rr.rerank("查询", [], top_k=3) == []


def test_get_reranker_default_score():
    """默认配置（RAG_RERANK=score）返回 ScoreReranker。"""
    old = settings.reranker
    settings.reranker = "score"
    try:
        rr = get_reranker()
        assert rr.name == "score"
    finally:
        settings.reranker = old


def test_get_reranker_local_without_model_degrades(monkeypatch):
    """配置 RAG_RERANK=local 但模型缺失 → 返回的仍是 LocalOnnxReranker（内部降级）。"""
    old = settings.reranker
    settings.reranker = "local"
    try:
        # 清缓存，确保新建
        import app.reranker as rr_mod

        rr_mod._reranker_cache.clear()
        monkeypatch.setattr(rr_mod, "RERANKER_MODEL_DIR", ROOT / "_models" / "__not_exist__")
        rr = get_reranker()
        assert rr.name == "local-cross"
        # 调用 rerank 应降级且不抛异常
        out = rr.rerank("查询", _cands(), top_k=2)
        assert out[0][1].id == 1
    finally:
        settings.reranker = old
        import app.reranker as rr_mod

        rr_mod._reranker_cache.clear()