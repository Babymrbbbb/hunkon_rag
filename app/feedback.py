"""用户反馈存储——生产库 feedback.py 的开源复刻。

能力等价：
  - SQLite 存储问答反馈（点赞/点踩）
  - 表 feedback：question / answer / citations(JSON) / rating(1=赞, -1=踩) / reason / created_at
  - 查询接口：按 rating 过滤、导出 golden set（JSON/CSV）

与生产库差异：
  - 生产库路径：KB/_meta/feedback.db
  - swan_rag 路径：data/feedback.db（相对路径，随仓库分发）

用法：
  from app.feedback import FeedbackStore
  store = FeedbackStore()
  store.add(question, answer, citations, rating=1, reason="")
  store.list(rating=1)
  store.export_golden_set()
"""
import json
import os
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "feedback.db"


class FeedbackStore:
    def __init__(self, db_path=None):
        self.db_path = str(db_path) if db_path else str(DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                rating INTEGER NOT NULL DEFAULT 1,
                reason TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)"
        )
        self._conn.commit()

    def add(self, question, answer, citations=None, rating=1, reason=""):
        """记录一条反馈。返回新记录 id。"""
        try:
            cur = self._conn.execute(
                """
                INSERT INTO feedback (question, answer, citations, rating, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question or "",
                    answer or "",
                    json.dumps(citations or [], ensure_ascii=False),
                    int(rating),
                    reason or "",
                    time.time(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception:
            return None

    def list(self, limit=200, rating=None):
        """查询反馈记录。rating: 1=赞, -1=踩, None=全部。"""
        try:
            sql = "SELECT id, question, answer, citations, rating, reason, created_at FROM feedback"
            params = []
            if rating is not None:
                sql += " WHERE rating = ?"
                params.append(int(rating))
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
            out = []
            for r in rows:
                try:
                    cites = json.loads(r[3] or "[]")
                except Exception:
                    cites = []
                out.append({
                    "id": r[0],
                    "question": r[1],
                    "answer": r[2],
                    "citations": cites,
                    "rating": r[4],
                    "reason": r[5],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[6])),
                })
            return out
        except Exception:
            return []

    def stats(self):
        """返回反馈统计。"""
        try:
            total = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            thumbs_up = self._conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = 1").fetchone()[0]
            thumbs_down = self._conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = -1").fetchone()[0]
            return {"total": total, "thumbs_up": thumbs_up, "thumbs_down": thumbs_down}
        except Exception:
            return {"total": 0, "thumbs_up": 0, "thumbs_down": 0}

    def export_golden_set(self, limit=500, min_rating=1):
        """导出 golden set（点赞过的问答对，供评测演进）。
        返回 [{question, answer, citations}]。
        """
        rows = self.list(limit=limit, rating=min_rating)
        return [{"question": r["question"], "answer": r["answer"],
                 "citations": r["citations"]} for r in rows]

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def main():
    """CLI：查看/导出反馈。"""
    import argparse
    import csv
    import sys

    ap = argparse.ArgumentParser(description="用户反馈存储工具")
    ap.add_argument("--stats", action="store_true", help="显示统计")
    ap.add_argument("--list", action="store_true", help="列出最近反馈")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--export", choices=["json", "csv"], help="导出 golden set")
    args = ap.parse_args()

    store = FeedbackStore()
    if args.stats:
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    if args.list:
        for r in store.list(limit=args.limit):
            mark = "👍" if r["rating"] == 1 else "👎"
            print(f"[{r['created_at']}] {mark} {r['question'][:60]}")
            if r["reason"]:
                print(f"    原因: {r['reason'][:80]}")
    if args.export:
        gs = store.export_golden_set()
        if args.export == "json":
            print(json.dumps(gs, ensure_ascii=False, indent=2))
        else:
            w = csv.writer(sys.stdout)
            w.writerow(["question", "answer", "citations"])
            for g in gs:
                w.writerow([g["question"], g["answer"], json.dumps(g["citations"], ensure_ascii=False)])
    store.close()


if __name__ == "__main__":
    main()