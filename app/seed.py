"""知识库自愈合：比对 corpus 内容哈希，库为空或与源不一致时自动重建索引。

动机：swan_rag 的向量索引（swan_rag.db 中的 document/chunk 表）是「源文件 → 灌库脚本」
手动生成的产物。此前曾出现「源 .md 已改正，却忘记重跑 seed_corpus.py，导致线上检索
仍返回旧错数据」的事故。本模块让应用在启动时自动检测源文件漂移并重建，杜绝该类复发。
"""
import hashlib
from pathlib import Path
from sqlmodel import Session, func, select

from app.config import settings
from app.embedder import get_embedder
from app.models import Chunk, Document, Meta
from app.rag import ingest

_HASH_KEY = "corpus_hash"
_FP_KEY = "embedder_fingerprint"


def embedder_fingerprint() -> str:
    """当前向量化实现的指纹（如 mock:256 / local:1024）。

    维度或模型变了，库里已存的向量就与新查询向量不可比（cosine 因长度不等恒为 0），
    因此指纹必须参与「是否需要重建」的判断，否则换模型后会静默失效。
    """
    return get_embedder().fingerprint


def corpus_dir() -> Path:
    """真实语料目录：来自配置（默认 data/corpus；可指向 D:\\WorkBuddy知识库\\wiki 等生产库）。"""
    return Path(settings.corpus_dir)


def corpus_hash() -> str:
    """对所有语料 *.md 内容求稳定哈希（递归子目录，按相对路径排序，避免目录遍历顺序差异）。"""
    h = hashlib.sha256()
    for f in sorted(corpus_dir().rglob("*.md")):
        h.update(f.read_bytes())
    return h.hexdigest()


def seed_corpus(session: Session) -> int:
    """把语料目录（含子目录）所有 *.md 逐篇 ingest 进向量库（不清空旧数据，仅追加）。"""
    count = 0
    for f in sorted(corpus_dir().rglob("*.md")):
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else f.stem
        ingest(session, title=title, content=text, source=str(f.relative_to(corpus_dir())))
        count += 1
    return count


def _clear_index(session: Session) -> None:
    """清空检索索引（document + chunk），保留 qa 审计日志。"""
    for row in session.exec(select(Chunk)).all():
        session.delete(row)
    for row in session.exec(select(Document)).all():
        session.delete(row)
    session.commit()


def rebuild_corpus(session: Session) -> int:
    """强制重建：清空旧索引（含已删除文档的残留 chunk）→ 重新灌库 → 写入最新哈希与向量指纹。"""
    _clear_index(session)
    n = seed_corpus(session)
    # upsert：先删除旧哈希再写入，避免重复启动时 UNIQUE 冲突（2026-10-30 压测时发现）
    for old in session.exec(select(Meta).where(Meta.key.in_([_HASH_KEY, _FP_KEY]))).all():
        session.delete(old)
    session.add(Meta(key=_HASH_KEY, value=corpus_hash()))
    session.add(Meta(key=_FP_KEY, value=embedder_fingerprint()))
    session.commit()
    return n


def reseed_if_needed(session: Session) -> bool:
    """库为空、源改动、或向量模型变更时自动重建；全部一致则跳过。"""
    if not settings.auto_seed:
        return False
    current_hash = corpus_hash()
    stored_hash = session.exec(select(Meta.value).where(Meta.key == _HASH_KEY)).first()
    stored_fp = session.exec(select(Meta.value).where(Meta.key == _FP_KEY)).first()
    doc_count = session.exec(select(func.count()).select_from(Document)).one()
    if stored_hash == current_hash and doc_count > 0 and stored_fp == embedder_fingerprint():
        return False
    rebuild_corpus(session)
    return True


def force_rebuild(session: Session) -> int:
    """评测/测试用：无视 auto_seed 与哈希一致，强制清空重灌（保证无脏数据残留）。"""
    return rebuild_corpus(session)
