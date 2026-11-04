"""反馈闭环测试：写入、查询、统计、导出 golden set。

背景：
  生产知识库 feedback.py（2026-11-03 P1-1），swan_rag 移植为 app/feedback.py。
  核心：对问答结果点赞/点踩，沉淀 golden set 供评测演进。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.feedback import FeedbackStore  # noqa: E402


def _store():
    db = Path(tempfile.mkdtemp()) / "feedback.db"
    return FeedbackStore(db_path=str(db))


def test_add_and_list():
    s = _store()
    sid = s.add("天鹅公主有多少家门店？", "47 家", [{"title": "门店网络"}], rating=1)
    assert sid is not None
    rows = s.list()
    assert len(rows) == 1
    assert rows[0]["question"] == "天鹅公主有多少家门店？"
    assert rows[0]["answer"] == "47 家"
    assert rows[0]["citations"][0]["title"] == "门店网络"
    assert rows[0]["rating"] == 1
    s.close()


def test_list_filter_by_rating():
    s = _store()
    s.add("问题A", "答案A", rating=1)
    s.add("问题B", "答案B", rating=-1)
    up = s.list(rating=1)
    down = s.list(rating=-1)
    assert len(up) == 1 and up[0]["question"] == "问题A"
    assert len(down) == 1 and down[0]["question"] == "问题B"
    s.close()


def test_stats():
    s = _store()
    assert s.stats() == {"total": 0, "thumbs_up": 0, "thumbs_down": 0}
    s.add("问题A", "答案A", rating=1)
    s.add("问题B", "答案B", rating=-1)
    s.add("问题C", "答案C", rating=1)
    st = s.stats()
    assert st["total"] == 3
    assert st["thumbs_up"] == 2
    assert st["thumbs_down"] == 1
    s.close()


def test_export_golden_set():
    s = _store()
    s.add("问题A", "答案A", [{"title": "来源1"}], rating=1)
    s.add("问题B", "答案B", rating=-1)  # 点踩的不应导出
    gs = s.export_golden_set()
    assert len(gs) == 1
    assert gs[0]["question"] == "问题A"
    assert gs[0]["citations"][0]["title"] == "来源1"
    s.close()


def test_reason_saved():
    s = _store()
    s.add("问题", "答案", rating=-1, reason="答案错误")
    rows = s.list()
    assert rows[0]["reason"] == "答案错误"
    s.close()