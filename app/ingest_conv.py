"""对话沉淀摄入器——生产库 ingest_conversations.py 的开源复刻。

能力等价：
  - 从 _inbox/conversation_log.jsonl 读取「有结论的对话摘要」
  - 每行 JSON：{"date","topic","conclusion","subkb_hint"}
  - 生成 .md 文件进语料目录（data/corpus），标记 done 避免重复摄入

与生产库差异：
  - 生产库写「对话沉淀/L3_原始资料」+ 11 子库分类 + 触发 daily_sync
  - swan_rag 写 data/corpus（语料目录），不分类（语料即全部知识）

用法：
  python -m app.ingest_conv [--dry-run]
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INBOX = ROOT / "_inbox" / "conversation_log.jsonl"
OUT_DIR = ROOT / "data" / "corpus"


def _norm_name(fn: str) -> str:
    """文件名归一化：去括号及内容、去分隔符/空格。"""
    stem = Path(fn).stem
    stem = re.sub(r"[（(].*?[)）]", "", stem)
    stem = re.sub(r"[-_\s]+", "", stem)
    return stem.lower()


def scan_existing_basenames() -> set:
    """预扫描语料目录，返回 {归一化名} 集合（防双写）。"""
    if not OUT_DIR.is_dir():
        return set()
    return {_norm_name(f.name) for f in OUT_DIR.glob("*.md")}


def main() -> int:
    ap = argparse.ArgumentParser(description="把对话沉淀摘要摄入知识库")
    ap.add_argument("--dry-run", action="store_true", help="只预览不写文件")
    args = ap.parse_args()

    if not INBOX.exists():
        print(f"NO_INBOX: {INBOX}")
        return 0

    lines = INBOX.read_text(encoding="utf-8", newline="").splitlines()
    pending = []
    pending_idx = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not obj.get("done"):
            pending.append(obj)
            pending_idx.append(i)

    if not pending:
        print("NO_PENDING")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = scan_existing_basenames()
    ingested = []
    skipped_dup = []

    for obj in pending:
        date = obj.get("date", datetime.date.today().isoformat())
        topic = obj.get("topic", "对话沉淀").replace("/", "_").replace("\\", "_")[:40]
        concl = obj.get("conclusion", "").strip()
        hint = obj.get("subkb_hint", "日志沉淀")
        fn = f"{date}_{topic}.md"
        full = OUT_DIR / fn

        # 双写防护：同名文件已存在 → 跳过
        if _norm_name(fn) in existing:
            print(f"SKIP(dup): {fn}", flush=True)
            obj["done"] = True
            skipped_dup.append(fn)
            continue
        if full.exists():
            print(f"SKIP(exists): {fn}", flush=True)
            obj["done"] = True
            skipped_dup.append(fn)
            continue

        content = [
            f"# {topic}",
            "",
            f"> 对话沉淀 · 自动摄入 | 日期: {date} | 建议子库: {hint}",
            "",
            "## 结论",
            concl or "(空)",
            "",
            "## 来源",
            "- 由 AI 对话自动沉淀(有结论才入库), 非源库手工资料。",
        ]
        text = "\n".join(content)

        if args.dry_run:
            print(f"[dry] {fn} -> {hint}")
        else:
            full.write_text(text + "\n", encoding="utf-8")
            obj["done"] = True
            ingested.append(fn)
            print(f"OK: {fn} -> {hint}", flush=True)

    if not args.dry_run:
        # 回写 inbox：按行下标精确标记 done
        new_lines = list(lines)
        touched = 0
        for i in pending_idx:
            try:
                o = json.loads(new_lines[i])
            except Exception:
                continue
            if not o.get("done"):
                o["done"] = True
                new_lines[i] = json.dumps(o, ensure_ascii=False)
                touched += 1
        INBOX.write_text("\n".join(new_lines).rstrip("\r\n") + "\n", encoding="utf-8", newline="")

        print(f"INGESTED {len(ingested)}, 跳过 {len(skipped_dup)}, 回写 done {touched}")

    return 0


if __name__ == "__main__":
    sys.exit(main())