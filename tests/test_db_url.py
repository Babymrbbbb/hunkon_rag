"""数据库 URL 构造测试：绝对路径不能让 SQLAlchemy 误判为相对路径。

背景（2026-11-03 CI 事故）：
    database.py 原先把 URL 硬拼为 `sqlite:///./{db_file}`。测试用 tempfile 传入
    绝对路径 /tmp/xxx/test.db 时，拼出 `sqlite:///.//tmp/xxx/test.db`，
    SQLAlchemy 解析为相对路径 `.//tmp/...`，目录不存在 → "unable to open database file"。
    Windows 下侥幸可用，Linux CI 必挂。
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import _sqlite_url  # noqa: E402


def test_relative_path_keeps_dot_prefix():
    """相对路径保持 sqlite:///./file 形式，落库到工作目录。"""
    assert _sqlite_url("hunkon_rag.db") == "sqlite:///./hunkon_rag.db"


def test_absolute_path_not_treated_as_relative():
    """绝对路径必须生成 sqlite:///<abs>，不能被加上 './' 前缀。"""
    abs_path = Path(tempfile.mkdtemp(prefix="hunkon_rag_url_test_")) / "t.db"
    url = _sqlite_url(str(abs_path))
    assert not url.startswith("sqlite:///./"), f"绝对路径被误加前缀: {url}"
    assert url.endswith(abs_path.as_posix()), f"URL 未包含绝对路径: {url}"
    # 去掉 scheme 后应是绝对路径（POSIX 以 / 开头，Windows 以盘符开头）
    tail = url[len("sqlite:///"):]
    assert tail.startswith("/") or (len(tail) > 1 and tail[1] == ":"), f"路径非绝对: {url}"
