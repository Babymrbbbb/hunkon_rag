"""每日自动同步——生产库 daily_sync.py 的开源复刻。

能力等价：
  1. 检测语料目录是否有新增/修改/删除
  2. 若有变化，触发 scripts/build_index.py 重建索引
  3. 生成当日同步报告（data/sync_report.txt）

与生产库差异：
  - 生产库源库是 D:\\Hunkon 知识库（绝对路径）+ 11 子库分类 + build_kb.py
  - hunkon_rag 源是 settings.corpus_dir（相对路径）+ scripts/build_index.py

用法：
  python -m app.daily_sync             # 检测 + 自动重建 + 报告
  python -m app.daily_sync --check     # 只检测
  python -m app.daily_sync --force     # 强制重建
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

REPORT = ROOT / "data" / "sync_report.txt"
STATE = ROOT / "data" / "sync_state.json"
TEXT_EXTS = {".md", ".txt", ".markdown", ".xlsx", ".docx", ".pdf"}


def corpus_dir() -> Path:
    d = Path(settings.corpus_dir)
    if not d.is_absolute():
        d = ROOT / d
    return d


def scan_source() -> dict:
    """扫描语料目录，返回 {rel: [mtime, size]}。"""
    snap = {}
    src = corpus_dir()
    if not src.is_dir():
        return None
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in TEXT_EXTS:
            continue
        rel = f.relative_to(src).as_posix()
        try:
            st = f.stat()
            snap[rel] = [round(st.st_mtime, 2), st.st_size]
        except OSError:
            pass
    return snap


def diff(old: dict, new: dict) -> tuple:
    old = old or {}
    new = new or {}
    added = [k for k in new if k not in old]
    removed = [k for k in old if k not in new]
    changed = [k for k in new if k in old and new[k] != old[k]]
    return added, changed, removed


def main() -> int:
    ap = argparse.ArgumentParser(description="每日自动同步语料 → 重建索引")
    ap.add_argument("--check", action="store_true", help="只检测不重建")
    ap.add_argument("--force", action="store_true", help="强制重建")
    args = ap.parse_args()

    new = scan_source()
    lines = [
        "=" * 50,
        "hunkon_rag · 每日同步报告",
        "运行时间: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "语料目录: " + str(corpus_dir()),
        "=" * 50,
    ]

    if new is None:
        lines.append("ERROR: 语料目录不存在")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("SRC_MISSING")
        return 1

    old = {}
    if STATE.exists():
        try:
            old = json.loads(STATE.read_text(encoding="utf-8")).get("files", {})
        except Exception:
            old = {}

    added, changed, removed = diff(old, new)
    stale = bool(added or changed or removed) or args.force

    lines += ["", "## 变更检测"]
    lines.append(f"  当前文件数: {len(new)}")
    lines.append(f"  新增: {len(added)}  修改: {len(changed)}  删除: {len(removed)}")
    if added[:10]:
        lines.append("  新增样例: " + "; ".join(added[:10]))
    if changed[:10]:
        lines.append("  修改样例: " + "; ".join(changed[:10]))
    if removed[:10]:
        lines.append("  删除样例: " + "; ".join(removed[:10]))

    if args.check:
        lines += ["", "## 模式: 仅检测 (--check)，未重建"]
        lines.append("结论: " + ("源库已更新，需要重建" if stale else "源库无变化，为最新"))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("STALE" if stale else "FRESH")
        return 0

    if not stale:
        lines += ["", "## 结论: 源库无变化，跳过重建"]
        lines.append("知识库保持最新，无需操作。")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("FRESH")
        return 0

    # 有变化，执行重建
    lines += ["", "## 执行自动重建"]
    try:
        r = subprocess.run(
            [sys.executable, "-m", "scripts.build_index", "--corpus", str(corpus_dir())],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode == 0:
            # 写新快照
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(
                json.dumps({"files": new, "ts": time.time()}, ensure_ascii=False),
                encoding="utf-8",
            )
            lines += ["  重建成功。", "  快照已刷新: " + str(STATE)]
            lines += ["", "## 本次同步完成 ✓"]
            print("REBUILT")
        else:
            lines.append(f"  重建失败 (exit={r.returncode})")
            lines.append((r.stderr or r.stdout or "").strip()[-2000:])
            print("REBUILD_FAILED")
    except Exception as e:
        lines.append(f"  重建异常: {e}")
        print("REBUILD_FAILED")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())