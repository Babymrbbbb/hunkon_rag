"""一致性校验测试：一致、缺失、多余、--force 重建。

背景：
  生产知识库 check_sync.py，hunkon_rag 移植为 app/check_sync.py。
  核心：校验语料目录与检索索引一致性。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from sqlmodel import Session

import app.check_sync as cs  # noqa: E402
from app.database import engine, init_db  # noqa: E402
from app.rag import ingest  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_env(monkeypatch):
    """把语料目录指向临时目录，并清空共享测试库（避免测试间串扰）。"""
    init_db()  # 先建表（conftest 只设 env，不建表）
    from app.models import Chunk, Document
    from sqlmodel import delete

    with Session(engine) as s:
        s.exec(delete(Chunk))
        s.exec(delete(Document))
        s.commit()

    tmp = Path(tempfile.mkdtemp())
    corpus = tmp / "corpus"
    corpus.mkdir()
    monkeypatch.setattr(cs.settings, "corpus_dir", str(corpus))
    yield {"corpus": corpus}


def _run(force=False, as_json=False):
    import sys as _sys

    old_argv = _sys.argv
    args = ["check_sync"]
    if force:
        args.append("--force")
    if as_json:
        args.append("--json")
    _sys.argv = args
    try:
        return cs.main()
    finally:
        _sys.argv = old_argv


def test_consistent(_tmp_env):
    """语料与库一致 → 报一致。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    with Session(engine) as s:
        ingest(s, title="A", content="# A\n内容", source="a.md")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run()  # 默认非 json 模式
    # 用 json 模式验证结构
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        _run(as_json=True)
    data = json.loads(buf2.getvalue())
    assert data["consistent"] is True
    assert data["corpus_count"] == 1
    assert data["db_count"] == 1


def test_missing_detected(_tmp_env):
    """语料有、库没有 → 报缺失。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    # 不灌库 → 必然缺失
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run()
    out = buf.getvalue()
    assert "一致: False" in out
    assert "缺失" in out


def test_extra_detected(_tmp_env):
    """库有、语料没有 → 报多余（残留）。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    with Session(engine) as s:
        ingest(s, title="A", content="# A\n内容", source="a.md")
        # 额外灌一篇语料里没有的
        ingest(s, title="B", content="# B\n内容", source="ghost.md")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run()
    out = buf.getvalue()
    assert "一致: False" in out
    assert "多余" in out


def test_force_rebuild(_tmp_env):
    """--force 应触发重建，使语料与库一致。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    # 先灌一篇不存在的 → 不一致
    with Session(engine) as s:
        ingest(s, title="OLD", content="旧内容", source="old.md")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run(force=True)
    out = buf.getvalue()
    assert "已触发重建" in out
    # 重建后应只剩语料里的 a.md
    import contextlib as ctx
    import io as _io

    buf2 = _io.StringIO()
    with ctx.redirect_stdout(buf2):
        _run(as_json=True)
    data = json.loads(buf2.getvalue())
    assert data["consistent"] is True
    assert data["db_count"] == 1
    assert data["extra"] == []