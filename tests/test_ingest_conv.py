"""对话沉淀摄入测试：读取 inbox、生成 md、标记 done、防双写。

背景：
  生产知识库 ingest_conversations.py，hunkon_rag 移植为 app/ingest_conv.py。
  核心：把有结论的对话摘要从 inbox 摄入知识库，避免重复摄入。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

import app.ingest_conv as ic  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_paths(monkeypatch):
    """把 INBOX 与 OUT_DIR 指向临时目录，避免污染仓库。"""
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(ic, "INBOX", tmp / "conversation_log.jsonl")
    monkeypatch.setattr(ic, "OUT_DIR", tmp / "corpus")
    yield


def _write_inbox(entries):
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    ic.INBOX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ingest_creates_md_and_marks_done():
    _write_inbox([
        {"date": "2026-11-04", "topic": "冰面包调研结论", "conclusion": "冰面包非在售", "subkb_hint": "产品研发"},
    ])
    # 直接调用 main 逻辑（模拟 argv 无 --dry-run）
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["ingest_conv"]
    try:
        rc = ic.main()
    finally:
        _sys.argv = old_argv
    assert rc == 0
    # 生成了 md 且内容正确
    files = list(ic.OUT_DIR.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "冰面包调研结论" in content
    assert "冰面包非在售" in content
    # inbox 已标记 done
    obj = json.loads(ic.INBOX.read_text(encoding="utf-8").splitlines()[0])
    assert obj.get("done") is True


def test_ingest_skips_duplicate():
    _write_inbox([
        {"date": "2026-11-04", "topic": "话题A", "conclusion": "结论A", "subkb_hint": "日志沉淀"},
        {"date": "2026-11-04", "topic": "话题A", "conclusion": "结论A", "subkb_hint": "日志沉淀"},
    ])
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["ingest_conv"]
    try:
        ic.main()
    finally:
        _sys.argv = old_argv
    # 只生成 1 个文件
    assert len(list(ic.OUT_DIR.glob("*.md"))) == 1


def test_dry_run_no_write():
    _write_inbox([
        {"date": "2026-11-04", "topic": "话题B", "conclusion": "结论B", "subkb_hint": "行业趋势"},
    ])
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["ingest_conv", "--dry-run"]
    try:
        rc = ic.main()
    finally:
        _sys.argv = old_argv
    assert rc == 0
    assert len(list(ic.OUT_DIR.glob("*.md"))) == 0, "dry-run 不应写文件"
    # inbox 不应标记 done
    obj = json.loads(ic.INBOX.read_text(encoding="utf-8").splitlines()[0])
    assert obj.get("done") is None or obj.get("done") is False


def test_no_inbox():
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["ingest_conv"]
    try:
        rc = ic.main()
    finally:
        _sys.argv = old_argv
    assert rc == 0