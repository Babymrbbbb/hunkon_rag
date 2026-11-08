"""向量指纹校验：换 embedding 模型（维度变化）必须自动重建索引。

为什么单独测：mock 256 维 → local 1024 维时，若库内仍是旧向量，
cosine 因长度不等恒返回 0（见 app/retriever.py:cosine），
表现为「检索突然变差却不报错」，属于最难排查的一类静默失效。
"""
import tempfile
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from app import seed as seed_mod
from app.models import Document, Meta
from app.seed import embedder_fingerprint, rebuild_corpus, reseed_if_needed

FP_KEY = "embedder_fingerprint"


def _fresh_session() -> Session:
    """独立临时库：不干扰共享测试库里的其它用例数据。"""
    tmp = Path(tempfile.mkdtemp(prefix="fp_")) / "t.db"
    engine = create_engine(
        f"sqlite:///{tmp.as_posix()}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _force_auto_seed(monkeypatch) -> None:
    # conftest 把 RAG_AUTO_SEED 设为 false，这里单独打开以测试自愈合逻辑
    monkeypatch.setattr(seed_mod.settings, "auto_seed", True, raising=False)


def test_rebuild_records_fingerprint(monkeypatch):
    _force_auto_seed(monkeypatch)
    with _fresh_session() as s:
        rebuild_corpus(s)
        stored = s.exec(select(Meta.value).where(Meta.key == FP_KEY)).first()
        assert stored == embedder_fingerprint()
        assert s.exec(select(Document)).all(), "重建后应有语料文档"


def test_reseed_skips_when_consistent(monkeypatch):
    _force_auto_seed(monkeypatch)
    with _fresh_session() as s:
        rebuild_corpus(s)
        assert reseed_if_needed(s) is False, "语料与向量指纹都未变，不应重复重建"


def test_reseed_rebuilds_on_fingerprint_change(monkeypatch):
    """模拟「换了向量模型但库里还是旧向量」——必须自动重建。"""
    _force_auto_seed(monkeypatch)
    with _fresh_session() as s:
        rebuild_corpus(s)
        for row in s.exec(select(Meta).where(Meta.key == FP_KEY)).all():
            s.delete(row)
        s.add(Meta(key=FP_KEY, value="mock:999"))
        s.commit()

        assert reseed_if_needed(s) is True, "指纹不一致时必须触发重建"
        stored = s.exec(select(Meta.value).where(Meta.key == FP_KEY)).first()
        assert stored == embedder_fingerprint()
