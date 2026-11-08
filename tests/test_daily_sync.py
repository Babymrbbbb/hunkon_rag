"""每日同步测试：变更检测、快照、--check 模式、报告生成。

背景：
  生产知识库 daily_sync.py，hunkon_rag 移植为 app/daily_sync.py。
  核心：检测语料变更 → 触发重建 → 生成报告。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

import app.daily_sync as ds  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_env(monkeypatch):
    """把语料目录、快照、报告指向临时目录。"""
    tmp = Path(tempfile.mkdtemp())
    corpus = tmp / "corpus"
    corpus.mkdir()
    monkeypatch.setattr(ds, "REPORT", tmp / "sync_report.txt")
    monkeypatch.setattr(ds, "STATE", tmp / "sync_state.json")
    monkeypatch.setattr(ds.settings, "corpus_dir", str(corpus))
    yield {"corpus": corpus, "tmp": tmp}


def _run(check=False, force=False):
    import sys as _sys

    old_argv = _sys.argv
    args = ["daily_sync"]
    if check:
        args.append("--check")
    if force:
        args.append("--force")
    _sys.argv = args
    try:
        return ds.main()
    finally:
        _sys.argv = old_argv


def test_fresh_when_no_change(_tmp_env):
    """初始无快照 → 视为有变化（新增），但 --check 只报告 STALE。"""
    ( _tmp_env["corpus"] / "a.md").write_text("# A\n内容", encoding="utf-8")
    rc = _run(check=True)
    assert rc == 0
    assert ds.STATE.exists() is False, "--check 不应写快照"
    report = ds.REPORT.read_text(encoding="utf-8")
    assert "新增: 1" in report


def test_stale_detection_on_add(_tmp_env):
    """有快照后新增文件 → --check 报 STALE。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    # 先跑一次非 check 触发重建（但重建会失败因为 corpus 是临时目录且脚本不存在，
    # 我们只需要它写快照。用 --force 让它尝试重建，然后手动写快照模拟成功。）
    ds.STATE.write_text(json.dumps({"files": ds.scan_source(), "ts": 0}), encoding="utf-8")

    # 新增文件
    (corpus / "b.md").write_text("# B\n内容", encoding="utf-8")
    rc = _run(check=True)
    assert rc == 0
    report = ds.REPORT.read_text(encoding="utf-8")
    assert "新增: 1" in report
    assert "需要重建" in report


def test_fresh_when_no_diff(_tmp_env):
    """快照与当前一致 → --check 报 FRESH。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    snap = ds.scan_source()
    ds.STATE.write_text(json.dumps({"files": snap, "ts": 0}), encoding="utf-8")
    rc = _run(check=True)
    assert rc == 0
    report = ds.REPORT.read_text(encoding="utf-8")
    assert "无变化" in report


def test_force_triggers_rebuild_attempt(_tmp_env, monkeypatch):
    """--force 应尝试触发重建（脚本不存在则失败但报告记录）。"""
    corpus = _tmp_env["corpus"]
    (corpus / "a.md").write_text("# A\n内容", encoding="utf-8")
    ds.STATE.write_text(json.dumps({"files": ds.scan_source(), "ts": 0}), encoding="utf-8")

    # monkeypatch subprocess.run 返回成功，避免真跑 build_index
    import subprocess

    class FakeResult:
        returncode = 0
        stdout = "构建完成"
        stderr = ""

    monkeypatch.setattr(ds.subprocess, "run", lambda *a, **kw: FakeResult())
    rc = _run(force=True)
    assert rc == 0
    assert ds.STATE.exists(), "重建成功应刷新快照"
    report = ds.REPORT.read_text(encoding="utf-8")
    assert "重建成功" in report


def test_missing_corpus(_tmp_env):
    """语料目录不存在 → 返回错误码。"""
    import shutil

    shutil.rmtree(_tmp_env["corpus"])
    rc = _run(check=True)
    assert rc == 1