"""一键构建索引：扫描语料目录（支持 md/xlsx/docx/pdf）→ 灌库 → 重建检索索引。

这是生产知识库 build_search_index.py + build_vector_index.py 的开源复刻等价物：
- 生产库：构建独立 JSON 倒排索引 + 向量索引文件
- hunkon_rag：语料直接灌入 SQLite（document/chunk 表），运行时由 Retriever 构建
  内存倒排 + 向量召回。因此「构建索引」= 扫描语料 → ingest → 刷新元信息。

用法：
  python -m scripts.build_index                          # 默认语料目录 data/corpus
  python -m scripts.build_index --corpus /path/to/docs   # 指定任意语料目录
  python -m scripts.build_index --corpus /path/to/docs --ext .md,.xlsx,.docx,.pdf

说明：
  - 支持格式由 app.extract.read_text 决定（md/txt/xlsx/docx/pdf）
  - ingest 直接写入 document/chunk；随后刷新 corpus 哈希与向量指纹，
    使 seed.reseed_if_needed 认为「库已最新」，避免启动时重复重建
  - 所有路径可用相对路径（相对项目根）或绝对路径
"""
import argparse
import sys
from pathlib import Path

# 允许 `python -m scripts.build_index` 与 `python scripts/build_index.py` 两种启动方式
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlmodel import Session  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import engine, init_db  # noqa: E402
from app.extract import read_text  # noqa: E402
from app.rag import ingest  # noqa: E402

DEFAULT_EXTS = {".md", ".txt", ".markdown", ".xlsx", ".docx", ".pdf"}


def iter_docs(corpus_dir: Path, exts: set):
    """遍历语料目录下所有支持的文件，产出 (abspath, relpath, title)。"""
    for f in sorted(corpus_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts:
            continue
        rel = f.relative_to(corpus_dir).as_posix()
        title = f.stem
        yield f, rel, title


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描语料目录 → 抽取文本 → 灌库 → 刷新索引元信息")
    ap.add_argument("--corpus", default=None, help="语料目录（默认 settings.corpus_dir）")
    ap.add_argument("--ext", default=None, help="逗号分隔扩展名，如 .md,.xlsx,.docx,.pdf")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus) if args.corpus else Path(settings.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    if not corpus_dir.is_dir():
        print(f"[错误] 语料目录不存在: {corpus_dir}")
        return 2

    exts = DEFAULT_EXTS
    if args.ext:
        exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
                for e in args.ext.split(",") if e.strip()}

    init_db()
    total = 0
    skipped = 0
    with Session(engine) as session:
        for f, rel, title in iter_docs(corpus_dir, exts):
            text = read_text(str(f))
            if not text.strip():
                print(f"  ⏭ 空内容跳过: {rel}")
                skipped += 1
                continue
            ingest(session, title=title, content=text, source=rel)
            total += 1
            print(f"  ✅ {rel} ({len(text)} 字)")
        # 刷新元信息：写入 corpus 哈希（用构建目录）与向量指纹，
        # 使 seed.reseed_if_needed 认为「库已最新」，避免启动时重复重建。
        from app.embedder import get_embedder
        from app.models import Meta
        from app.seed import _FP_KEY, _HASH_KEY, _invalidate_retrieval_cache
        from sqlmodel import select

        for old in session.exec(select(Meta).where(Meta.key.in_([_HASH_KEY, _FP_KEY]))).all():
            session.delete(old)
        session.add(Meta(key=_HASH_KEY, value=f"manual:{corpus_dir.as_posix()}"))
        session.add(Meta(key=_FP_KEY, value=get_embedder().fingerprint))
        session.commit()
        _invalidate_retrieval_cache()

    print(f"\n构建完成: {total} 篇入库, {skipped} 篇跳过")
    return 0


if __name__ == "__main__":
    sys.exit(main())