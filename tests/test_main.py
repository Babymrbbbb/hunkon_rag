"""接口与评测测试：全程使用 MockProvider（离线、可复现、CICD 友好）。

运行：
    pytest -q
"""
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import app 之前设置环境变量（settings 在导入时即读取）
_TMP = tempfile.mkdtemp(prefix="hunkon_rag_test_")
os.environ["RAG_RAG_DB_FILE"] = os.path.join(_TMP, "test.db")
os.environ["RAG_API_KEY"] = "test-secret-key"
os.environ["RAG_LLM_PROVIDER"] = "mock"
os.environ["RAG_EMBEDDER_PROVIDER"] = "mock"
# 测试自行负责灌库（setup_module），关闭启动自愈合以免重复入库
os.environ["RAG_AUTO_SEED"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session  # noqa: E402

from app.database import engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.rag import ingest  # noqa: E402

HEADERS = {"X-API-Key": "test-secret-key"}


def _ingest_corpus() -> None:
    corpus_dir = ROOT / "data" / "corpus"
    for f in sorted(corpus_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else f.stem
        ingest_session = Session(engine)
        try:
            ingest(ingest_session, title=title, content=text, source=f.name)
        finally:
            ingest_session.close()


def setup_module(module):
    init_db()
    _ingest_corpus()


client = TestClient(app)


def test_health():
    r = client.get("/health", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["llm"] == "mock"


def test_missing_api_key_returns_401():
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == 401


def test_wrong_api_key_returns_401():
    r = client.post("/chat", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_ingest_and_chat():
    r = client.post(
        "/ingest",
        json={"title": "测试门店", "content": "演示品牌 在A 市有 40 家门店，全品牌共 12 家，主打黑金可颂。"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/chat", json={"question": "演示品牌 有多少家门店？"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["citations"]
    assert any("12" in c["snippet"] or "门店" in c["title"] for c in body["citations"])


def test_metrics():
    r = client.get("/metrics", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["chat_requests"] >= 1


def test_eval_retrieval_hits():
    r = client.post("/eval", headers=HEADERS)
    assert r.status_code == 200
    rep = r.json()
    assert rep["total"] >= 8
    # 离线 mock 下，召回命中率应稳定 100%（检索与切分正确性的自动证明）。
    # 阈值与 CI 门禁对齐（0.95）：此前写 0.75 会在真实回归时放行（2026-11-03 收紧）。
    assert rep["retrieval_hit_rate"] >= 0.95


def test_validation_422():
    r = client.post("/ingest", json={"content": "缺少 title"}, headers=HEADERS)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == 422


# ---------- /search 纯检索（2026-11-03 新增） ----------
# 这组测试守的是三件事：鉴权不因新增路由而漏、返回的是可作「出处」的完整原文、
# 以及它确实没走 LLM（否则和 /chat 分成两条路就没意义）。


def test_search_requires_api_key():
    r = client.post("/search", json={"query": "门店"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == 401


def test_search_returns_hits_with_source():
    r = client.post("/search", json={"query": "演示品牌 有多少家门店？"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "演示品牌 有多少家门店？"
    assert body["hits"], "检索必须有命中，空结果说明索引没建好"
    first = body["hits"][0]
    assert first["rank"] == 1
    assert first["title"] and first["title"] != "未知", "出处标题不能为空或兜底值"
    assert first["text"].strip(), "原文不能为空"
    assert isinstance(first["score"], float)
    # rank 连续递增，前端可直接按序渲染
    assert [h["rank"] for h in body["hits"]] == list(range(1, len(body["hits"]) + 1))


def test_search_returns_untruncated_text():
    """/search 的用途是给出处，chunk 原文不能被截断成片段。"""
    from sqlmodel import select

    from app.models import Chunk

    r = client.post("/search", json={"query": "报损", "top_k": 3}, headers=HEADERS)
    assert r.status_code == 200
    top = r.json()["hits"][0]
    with Session(engine) as s:
        row = s.exec(
            select(Chunk).where(
                Chunk.document_id == top["document_id"], Chunk.seq == top["seq"]
            )
        ).first()
    assert row is not None
    assert row.text == top["text"], "返回文本与库内原文不一致（被截断或改写）"


def test_search_skips_llm():
    """/search 不得计入 chat_requests —— 它走的是纯检索，没调模型。"""
    before = client.get("/metrics", headers=HEADERS).json()["chat_requests"]
    r = client.post("/search", json={"query": "加盟要多少钱"}, headers=HEADERS)
    assert r.status_code == 200
    after = client.get("/metrics", headers=HEADERS).json()["chat_requests"]
    assert after == before, "/search 不应产生 chat 计数"


def test_search_top_k_bounds():
    for bad in (0, 21):
        r = client.post("/search", json={"query": "门店", "top_k": bad}, headers=HEADERS)
        assert r.status_code == 422, f"top_k={bad} 应被拒绝"
        assert r.json()["error"]["code"] == 422


def test_search_empty_query_422():
    r = client.post("/search", json={"query": ""}, headers=HEADERS)
    assert r.status_code == 422
