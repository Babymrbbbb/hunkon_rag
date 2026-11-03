"""数据库连接与会话：sqlite + SQLModel，Depends 注入。

并发压测优化（2026-10-30）：SQLite 默认 rollback journal 模式在并发写时
单写者串行 + 每次 commit 都 fsync，导致 /chat 高并发下 QA 审计落库互相锁等待。
这里改为 WAL + synchronous=NORMAL + busy_timeout：
- WAL：读写不互斥，写事务追加到 wal 文件，大幅降低锁竞争；
- synchronous=NORMAL：WAL 模式下崩溃不损坏库，只是可能丢最近提交（审计可接受）；
- busy_timeout=5000：并发写排队而非立即报 database is locked。
"""
import os
from pathlib import Path

from sqlalchemy import event
from sqlmodel import SQLModel, create_engine, Session

from app.config import settings


def _sqlite_url(db_file: str) -> str:
    """构造 sqlite URL，正确区分绝对路径与相对路径。

    修复（2026-11-03）：此前硬拼 "sqlite:///./{file}"，当 db_file 是绝对路径
    （如测试用 tempfile 得到的 /tmp/xxx/test.db）时会变成 sqlite:///.//tmp/xxx，
    被 SQLAlchemy 当作相对路径 .//tmp/xxx，目录不存在直接报
    "unable to open database file"。Windows 下恰好侥幸可用，Linux CI 必挂。
    """
    p = Path(db_file)
    if p.is_absolute():
        # Windows 需转成 C:/a/b 形式；POSIX 保持 /tmp/x，最终为 sqlite:////tmp/x
        return "sqlite:///" + p.as_posix()
    return "sqlite:///./" + p.as_posix()


sqlite_url = _sqlite_url(settings.rag_db_file)
engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """每个连接建立时设置 SQLite 并发优化 PRAGMA。"""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def init_db() -> None:
    """建表（仅本地兜底；生产应使用 Alembic 迁移）。"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI 依赖：每次请求一个 Session，自动关闭。"""
    with Session(engine) as session:
        yield session
