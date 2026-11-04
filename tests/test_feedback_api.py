"""反馈 API 测试：POST /api/feedback、GET /api/feedback/stats、/list。

注意：FeedbackStore 默认写 data/feedback.db（项目内），测试会污染仓库。
因此这里 monkeypatch app.feedback.DB_PATH 到临时目录。
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from app.main import app  # noqa: E402

HEADERS = {"X-API-Key": "test-secret-key"}


@pytest.fixture(autouse=True)
def _tmp_feedback_db(monkeypatch):
    """把 FeedbackStore 的默认 DB 路径指向临时目录，避免污染仓库。"""
    import app.feedback as fb

    tmp = Path(tempfile.mkdtemp()) / "feedback.db"
    monkeypatch.setattr(fb, "DB_PATH", tmp)
    # main.py 已 import FeedbackStore（类），其 __init__ 默认参数在 import 时已绑定 DB_PATH，
    # 因此还需要让 main 里实例化时读到新路径——通过 monkeypatch 模块属性即可，
    # 因为 FeedbackStore.__init__ 里 `db_path or str(DB_PATH)` 的 DB_PATH 是模块级全局。
    # 但 main.py 里 `from app.feedback import FeedbackStore` 引用的是类，类方法内
    # 访问的 DB_PATH 是 app.feedback 模块属性，monkeypatch 后生效。
    yield


def test_feedback_post_and_stats():
    c = TestClient(app)
    r = c.post(
        "/api/feedback",
        json={
            "question": "天鹅公主有多少家门店？",
            "answer": "47 家",
            "citations": [{"title": "门店网络"}],
            "rating": 1,
        },
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    st = c.get("/api/feedback/stats", headers=HEADERS).json()
    assert st["total"] == 1
    assert st["thumbs_up"] == 1
    assert st["thumbs_down"] == 0


def test_feedback_list_filter():
    c = TestClient(app)
    c.post("/api/feedback", json={"question": "Q1", "answer": "A1", "rating": 1}, headers=HEADERS)
    c.post("/api/feedback", json={"question": "Q2", "answer": "A2", "rating": -1}, headers=HEADERS)

    all_rows = c.get("/api/feedback/list", headers=HEADERS).json()
    assert len(all_rows) == 2

    up = c.get("/api/feedback/list", params={"rating": 1}, headers=HEADERS).json()
    assert len(up) == 1 and up[0]["question"] == "Q1"

    down = c.get("/api/feedback/list", params={"rating": -1}, headers=HEADERS).json()
    assert len(down) == 1 and down[0]["question"] == "Q2"