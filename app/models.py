"""数据库表模型（table=True）。注意：请求/响应体请走 schemas.py，避免表模型误当入参。"""
import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    source: str = ""
    content: str
    created_at: str = Field(default_factory=_now)


class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    seq: int = 0
    text: str
    # 向量以 JSON 字符串存储（demo 用 sqlite；生产可换 pgvector）
    embedding: str = ""


class QA(SQLModel, table=True):
    """记录历史问答，便于审计与评测复盘。"""
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str
    citations: str = ""  # JSON: [{document_id,title,snippet}]
    created_at: str = Field(default_factory=_now)


class Meta(SQLModel, table=True):
    """键值元表：当前用于保存 corpus 内容哈希，支撑索引自愈合。"""
    key: str = Field(primary_key=True)
    value: str = ""
