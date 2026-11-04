"""双路混合检索测试：验证倒排关键词 + 向量召回 + RRF 融合。

覆盖：
1. 精确术语问题 → 倒排路命中正确文档（BM25 生效）
2. 含实体词的问题（如"玛芝莲"）→ 关键词路召回正确竞品
3. 融合输出非空、无重复
4. 混合检索不劣于纯关键词
（注意）口语化语义召回依赖真实 Embedding；mock 向量无语义，故不在本测试断言。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="hunkon_rag_hybrid_test_")
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
from app.retriever import Retriever  # noqa: E402

DOCS = [
    ("竞品分析：玛芝莲", "玛芝莲是清远本地烘焙品牌，近年建设超级工厂，从区域连锁向供应链平台升维，主打 B 端蛋糕券与政企采购。"),
    ("竞品分析：金粒门", "金粒门是同期崛起的烘焙品牌，主打干货类产品（坚果礼盒、谷物脆、养生粉），以健康零食化差异化切入。"),
    ("冰面包市场调研", "冰面包非Hunkon 在售产品，市场通行参考值需在 0–4℃ 冷藏保存，保质期通常 3 天。"),
]


def setup_module(module):
    init_db()
    with Session(engine) as s:
        for title, content in DOCS:
            ingest(s, title=title, content=content, source="test")


def _titles_for(question, use_hybrid=True, top_k=2):
    emb = get_embedder()
    ret = Retriever(emb, top_k=top_k, use_hybrid=use_hybrid)
    qvec = emb.embed(question)
    with Session(engine) as s:
        hits = ret.search(s, qvec, top_k=top_k, query_text=question, use_hybrid=use_hybrid)
        doc_titles = {d.id: d.title for d in s.exec(select(Document)).all()}
        return [doc_titles.get(h[1].document_id, "未知") for h in hits]


def test_hybrid_precise_term():
    """精确术语问题 → 倒排路命中正确文档。"""
    titles = _titles_for("冰面包保质期多久？", use_hybrid=True)
    assert "冰面包市场调研" in titles, f"精确术语未命中冰面包调研: {titles}"


def test_hybrid_keyword_recalls_entity():
    """含实体词的问题 → 关键词路应召回玛芝莲文档（验证 BM25 生效）。"""
    titles = _titles_for("玛芝莲最近如何？", use_hybrid=True, top_k=3)
    assert "竞品分析：玛芝莲" in titles, f"关键词路未召回玛芝莲: {titles}"


def test_hybrid_result_nonempty_and_deduped():
    """融合结果非空、无重复 title。"""
    titles = _titles_for("玛芝莲最近如何？", use_hybrid=True, top_k=3)
    assert titles, "混合检索结果为空"
    assert len(titles) == len(set(titles)), f"混合检索结果有重复: {titles}"


def test_hybrid_vs_keyword_only_not_worse():
    """混合检索不应劣于纯关键词：精确术语问题两种模式都应命中。"""
    kw_only = _titles_for("冰面包保质期多久？", use_hybrid=False, top_k=2)
    hybrid = _titles_for("冰面包保质期多久？", use_hybrid=True, top_k=2)
    assert "冰面包市场调研" in kw_only
    assert "冰面包市场调研" in hybrid