"""检索契约测试：锁死 (score, chunk_id) 返回顺序，防 RRF 融合静默失效。

背景（2026-11-03 生产 bug）：
    InvertedIndex.search() 实际返回 (chunk_id, score)，但 rrf_fuse 按 (score, cid) 解包，
    导致关键词路的 chunk_id 与 score 互换。浮点 score 被当成 chunk_id 写入融合表，
    最终在 `if cid in by_id` 处被全部丢弃且不报错——混合检索静默退化为纯向量检索。
    当时 22 个测试全绿，因为向量路恰好也能召回目标文档，掩盖了问题。

本文件用三类断言锁死契约：
1. InvertedIndex.search 返回 (score: float, chunk_id: int)；
2. rrf_fuse 对顺序颠倒的入参也能正确融合（防御性）；
3. 端到端：关键词高分文档必须出现在混合检索结果中（本 bug 的直接回归）。
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="hunkon_rag_rrf_test_")
os.environ["RAG_RAG_DB_FILE"] = os.path.join(_TMP, "test.db")
os.environ["RAG_API_KEY"] = "test-secret-key"
os.environ["RAG_LLM_PROVIDER"] = "mock"
os.environ["RAG_EMBEDDER_PROVIDER"] = "mock"
os.environ["RAG_AUTO_SEED"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import Session, select  # noqa: E402

from app.database import engine, init_db  # noqa: E402
from app.embedder import get_embedder  # noqa: E402
from app.models import Document  # noqa: E402
from app.rag import ingest  # noqa: E402
from app.retriever import InvertedIndex, Retriever, get_cached_chunks, rrf_fuse  # noqa: E402

DOCS = [
    ("竞品分析：金粒门", "金粒门是同期崛起的烘焙品牌，主打干货类产品（坚果礼盒、谷物脆、养生粉），以健康零食化差异化切入。"),
    ("竞品分析：玛芝莲", "玛芝莲是清远本地烘焙品牌，建设超级工厂，向供应链平台升维，覆盖面包蛋糕西点与节令食品。"),
    ("冰面包市场调研", "冰面包非在售，市场通行参考值需 0-4 度冷藏保存，保质期 3 天。"),
]


def setup_module(module):
    init_db()
    with Session(engine) as s:
        for title, content in DOCS:
            ingest(s, title=title, content=content, source="test")


def test_inverted_index_returns_score_then_id():
    """InvertedIndex.search 必须返回 (score: float, chunk_id: int)。"""
    with Session(engine) as s:
        chunks, index, _by_id = get_cached_chunks(s)
    ranked = index.search("金粒门主打什么", top_k=3)
    assert ranked, "关键词路无结果"
    for score, cid in ranked:
        assert isinstance(score, float), f"第一元素应为 float score，实际 {type(score)}: {(score, cid)}"
        assert isinstance(cid, int), f"第二元素应为 int chunk_id，实际 {type(cid)}: {(score, cid)}"


def test_rrf_fuse_tolerates_reversed_input():
    """rrf_fuse 对 (chunk_id, score) 颠倒入参也应正确融合（不产生浮点 key）。"""
    normal = [(0.9, 1), (0.8, 2)]
    reversed_ = [(1, 0.9), (2, 0.8)]
    a = rrf_fuse([normal], k=60, top_k=2)
    b = rrf_fuse([reversed_], k=60, top_k=2)
    assert [cid for _s, cid in a] == [1, 2]
    assert [cid for _s, cid in b] == [1, 2], f"颠倒入参未被纠正: {b}"
    # 所有返回的 cid 必须是 int
    for _s, cid in a + b:
        assert isinstance(cid, int)


def test_keyword_high_score_doc_reaches_hybrid_result():
    """回归：关键词路的命中必须真实进入混合检索结果。

    修复前：关键词结果因顺序互换被整体丢弃（浮点 key 被 by_id 过滤），
    混合检索结果里不会出现任何「仅由关键词路召回的 chunk」。

    为避免依赖具体 chunk id（多个测试模块共用同一进程内 settings/数据库，
    id 会随执行顺序漂移），这里用集合交集断言：
    index.search 的 Top3 与混合检索 Top6 至少有一个 chunk 重合。
    """
    emb = get_embedder()
    ret = Retriever(emb, top_k=6, use_hybrid=True)
    q = "金粒门主打什么品类的产品？"
    with Session(engine) as s:
        _chunks, index, _by_id = get_cached_chunks(s)
        kw_ids = {cid for _score, cid in index.search(q, top_k=3)}
        assert kw_ids, "关键词路无召回"
        hits = ret.search(s, emb.embed(q), top_k=6, query_text=q, use_hybrid=True)
        got_ids = {ch.id for _sc, ch in hits}
        assert kw_ids & got_ids, (
            f"关键词路 Top3 {kw_ids} 与混合检索结果 {got_ids} 完全无交集——"
            "关键词路疑似被 RRF 顺序错位整体丢弃"
        )


def test_all_returned_chunks_are_real():
    """混合检索结果不能包含不存在的 chunk（旧 bug 会返回浮点 key 后被过滤，导致结果数变少）。"""
    emb = get_embedder()
    ret = Retriever(emb, top_k=3, use_hybrid=True)
    q = "冰面包市场调研的参考保质期多久？"
    with Session(engine) as s:
        chunks, _index, by_id = get_cached_chunks(s)
        hits = ret.search(s, emb.embed(q), top_k=3, query_text=q, use_hybrid=True)
        assert hits, "混合检索无结果"
        for _sc, ch in hits:
            assert ch.id in by_id, f"返回了不存在的 chunk: {ch.id}"
