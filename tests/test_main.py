"""接口与评测测试：全程使用 MockProvider（离线、可复现、CICD 友好）。

运行：
    pytest -q
"""
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import app 之前设置环境变量（settings 在导入时即读取）
_TMP = tempfile.mkdtemp(prefix="swan_rag_test_")
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
        json={"title": "测试门店", "content": "天鹅公主在清远有 40 家门店，全品牌共 47 家，主打黑金可颂。"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/chat", json={"question": "天鹅公主有多少家门店？"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["citations"]
    assert any("47" in c["snippet"] or "门店" in c["title"] for c in body["citations"])


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
