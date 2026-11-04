"""语义缓存模块——生产库 semantic_cache.py 的开源复刻。

能力等价：
  - SQLite 存储语义等价的历史问答
  - 命中判定：查询向量与缓存向量余弦相似度 >= 阈值（默认 0.95）
  - TTL 过期（默认 7 天），命中计数供分析
  - 可用环境变量关闭（关闭时零开销）

与生产库的差异（适配 hunkon_rag 配置体系）：
  - 生产库用 os.environ 直接读 KB_CACHE_THRESHOLD / KB_CACHE_TTL / KB_SEMANTIC_CACHE
  - hunkon_rag 用 pydantic-settings（RAG_ 前缀），因此这里通过 settings 读取，
    同时保留环境变量直读作为兜底（方便不开 FastAPI 的脚本使用）

用法：
  from app.cache import SemanticCache
  cache = SemanticCache()
  hit = cache.get(query_vec, question)   # -> (answer, citations, hit_count, cached_q) | None
  cache.put(query_vec, question, answer, citations)
"""
import json
import math
import os
import sqlite3
import time
from pathlib import Path

# 默认缓存路径：项目根 / data / semantic_cache.db（随仓库分发，gitignore 由用户决定）
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "semantic_cache.db"


def _cosine(a, b) -> float:
    """余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """SQLite 语义缓存。"""

    def __init__(self, db_path=None, threshold=None, ttl=None, enabled=None):
        self.db_path = str(db_path) if db_path else str(DEFAULT_DB)
        # 阈值/TTL：优先显式参数，其次 settings（RAG_ 前缀），最后环境变量直读
        self.threshold = (
            threshold
            if threshold is not None
            else self._env_float("RAG_CACHE_THRESHOLD", 0.95)
        )
        self.ttl = ttl if ttl is not None else self._env_int("RAG_CACHE_TTL", 7 * 24 * 3600)
        self.enabled = (
            enabled
            if enabled is not None
            else os.environ.get("RAG_SEMANTIC_CACHE", "1").lower()
            not in ("0", "false", "no", "off")
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        if self.enabled:
            self._init_db()

    @staticmethod
    def _env_float(name, default):
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_int(name, default):
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    def _init_db(self):
        """建表 + 清理过期条目。"""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_vec TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                last_hit_at REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_created ON semantic_cache(created_at)"
        )
        self._conn.commit()
        # 清理过期条目
        try:
            cutoff = time.time() - self.ttl
            self._conn.execute("DELETE FROM semantic_cache WHERE created_at < ?", (cutoff,))
            self._conn.commit()
        except Exception:
            pass

    def get(self, query_vec, question=None):
        """查找语义等价的缓存条目。命中返回 (answer, citations, hit_count, cached_q)。"""
        if not self.enabled or not query_vec:
            return None
        try:
            rows = self._conn.execute(
                "SELECT query_vec, question, answer, citations, hit_count, id, created_at FROM semantic_cache"
            ).fetchall()
        except Exception:
            return None
        best = None
        best_score = self.threshold
        now = time.time()
        for vec_text, cached_q, answer, citations, hit_count, cid, created_at in rows:
            # TTL 检查：过期条目不返回（与 _init_db 的清理互补，保证运行中也会过期）
            if now - created_at > self.ttl:
                continue
            try:
                vec = json.loads(vec_text)
            except Exception:
                continue
            score = _cosine(query_vec, vec)
            if score >= best_score:
                best_score = score
                best = (answer, citations, hit_count, cid, cached_q)
        if best is None:
            return None
        answer, citations, hit_count, cid, cached_q = best
        # 更新命中计数
        try:
            self._conn.execute(
                "UPDATE semantic_cache SET hit_count = ?, last_hit_at = ? WHERE id = ?",
                (hit_count + 1, time.time(), cid),
            )
            self._conn.commit()
        except Exception:
            pass
        return answer, citations, hit_count, cached_q

    def put(self, query_vec, question, answer, citations):
        """写入缓存条目。"""
        if not self.enabled or not query_vec or not answer:
            return
        try:
            self._conn.execute(
                """
                INSERT INTO semantic_cache
                    (query_vec, question, answer, citations, created_at, hit_count, last_hit_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    json.dumps(query_vec, ensure_ascii=False),
                    question,
                    answer,
                    json.dumps(citations, ensure_ascii=False),
                    time.time(),
                    time.time(),
                ),
            )
            self._conn.commit()
        except Exception:
            pass

    def stats(self):
        """返回缓存统计。"""
        if not self.enabled:
            return {"enabled": False, "entries": 0, "total_hits": 0}
        try:
            n = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
            hits = self._conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM semantic_cache"
            ).fetchone()[0]
            return {"enabled": True, "entries": n, "total_hits": hits}
        except Exception:
            return {"enabled": True, "entries": 0, "total_hits": 0}

    def clear(self):
        """清空缓存（运维用）。"""
        if not self.enabled:
            return
        try:
            self._conn.execute("DELETE FROM semantic_cache")
            self._conn.commit()
        except Exception:
            pass

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None