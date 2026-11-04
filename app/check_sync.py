"""一致性校验——生产库 check_sync.py 的开源复刻。

能力等价：
  - 扫描语料目录，与检索库（document/chunk 表）比对
  - 输出「缺失 / 多余 / 一致」诊断报告
  - 可选触发重建（--force 调用 seed.force_rebuild）

与生产库差异：
  - 生产库与 _meta/sync_state.json 快照比对 + 调 build_kb.py 重建
  - hunkon_rag 与 SQLite 中 Document 表的 source 字段比对（更直接）
  - 重建走 seed.force_rebuild（已有自愈合机制）

用法：
  python -m app.check_sync [--force] [--json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlmodel import Session, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import engine, init_db  # noqa: E402
from app.models import Document  # noqa: E402

TEXT_EXTS = {".md", ".txt", ".markdown", ".xlsx", ".docx", ".pdf"}


def corpus_dir() -> Path:
    d = Path(settings.corpus_dir)
    if not d.is_absolute():
        d = ROOT / d
    return d


def scan_corpus() -> dict:
    """扫描语料目录，返回 {source_rel: title}。"""
    src = corpus_dir()
    out = {}
    if not src.is_dir():
        return out
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in TEXT_EXTS:
            continue
        out[f.relative_to(src).as_posix()] = f.stem
    return out


def db_sources(session: Session) -> dict:
    """读取库中 Document 的 source 字段，返回 {source: title}。"""
    rows = session.exec(select(Document)).all()
    return {d.source: d.title for d in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="校验语料目录与检索索引一致性")
    ap.add_argument("--force", action="store_true", help="不一致时触发重建")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    init_db()
    corpus = scan_corpus()
    with Session(engine) as session:
        db = db_sources(session)
        db_sources_set = set(db.keys())
        corpus_set = set(corpus.keys())
        missing = sorted(corpus_set - db_sources_set)  # 语料有、库没有
        extra = sorted(db_sources_set - corpus_set)    # 库有、语料没有（残留）
        consistent = not missing and not extra

        if args.json:
            print(json.dumps({
                "consistent": consistent,
                "corpus_count": len(corpus),
                "db_count": len(db),
                "missing": missing,
                "extra": extra,
            }, ensure_ascii=False, indent=2))
            return 0

        print(f"语料文件数: {len(corpus)}")
        print(f"库中文档数: {len(db)}")
        print(f"一致: {consistent}")
        if missing:
            print(f"缺失（语料有、库没有）: {len(missing)}")
            for m in missing[:10]:
                print(f"  ❌ {m}")
        if extra:
            print(f"多余（库有、语料没有）: {len(extra)}")
            for e in extra[:10]:
                print(f"  ⚠️ {e}")

        # --force：不一致时触发重建（清空重灌 + 刷新缓存）
        if not consistent and args.force:
            from app.seed import force_rebuild

            n = force_rebuild(session)
            print(f"\n已触发重建: {n} 篇重新入库")
        elif not consistent and not args.force:
            print("\n提示: 加 --force 可触发重建（清空重灌）")

    return 0


if __name__ == "__main__":
    sys.exit(main())