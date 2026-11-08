"""灌库脚本（命令行）。

应用启动已自动比对 corpus 哈希并自愈合重建（见 app/seed.reseed_if_needed），
因此日常编辑源文件后无需手动灌库。本脚本用于 CI / 本地强制重建，或 auto_seed 关闭时手动初始化。
用法：
    python seed_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlmodel import Session

from app.database import engine, init_db
from app.seed import rebuild_corpus


def main() -> None:
    init_db()
    with Session(engine) as s:
        n = rebuild_corpus(s)
    print(f"已强制重建索引 {n} 篇文档（位于 data/corpus）")


if __name__ == "__main__":
    main()
