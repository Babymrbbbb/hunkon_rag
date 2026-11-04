"""语义缓存测试：写入、命中、阈值、TTL、禁用、统计、清空。

背景：
  生产知识库 semantic_cache.py（2026-11-03 P0-3），swan_rag 移植为 app/cache.py。
  核心：对语义等价的重复问题直接返回历史答案，跳过 LLM 调用。
"""
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cache import SemanticCache, _cosine  # noqa: E402


def _vec(dim: int = 8, seed: int = 1) -> list:
    """构造确定性单位向量（不同 seed 得到不同方向）。"""
    v = [math.sin(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def test_cosine():
    a = [1.0, 0.0]
    b = [1.0, 0.0]
    c = [0.0, 1.0]
    assert abs(_cosine(a, b) - 1.0) < 1e-9
    assert abs(_cosine(a, c)) < 1e-9
    assert _cosine([], []) == 0.0
    assert _cosine([1.0], [1.0, 2.0]) == 0.0


def test_put_and_get_hit():
    db = Path(tempfile.mkdtemp()) / "cache.db"
    c = SemanticCache(db_path=str(db), threshold=0.9, enabled=True)
    v = _vec(seed=1)
    c.put(v, "问题A", "答案A", [{"title": "来源1"}])
    hit = c.get(v, "问题A")
    assert hit is not None, "写入后应能命中"
    answer, citations, hit_count, cached_q = hit
    assert answer == "答案A"
    assert json.loads(citations)[0]["title"] == "来源1"
    assert cached_q == "问题A"
    # 再次命中应计数 +1
    c.get(v, "问题A")
    assert c.stats()["total_hits"] == 2, f"命中计数未累加: {c.stats()}"
    c.close()


def test_near_duplicate_hit_with_threshold():
    db = Path(tempfile.mkdtemp()) / "cache.db"
    c = SemanticCache(db_path=str(db), threshold=0.95, enabled=True)
    v1 = _vec(seed=1)
    v2 = [x + 0.01 * math.sin(i) for i, x in enumerate(v1)]  # 近似向量
    norm = math.sqrt(sum(x * x for x in v2))
    v2 = [x / norm for x in v2]
    c.put(v1, "问题", "答案", "[]")
    # 相似度应高于阈值
    assert _cosine(v1, v2) > 0.95
    hit = c.get(v2, "问题")
    assert hit is not None, "近似语义应命中"
    c.close()


def test_dissimilar_no_hit():
    db = Path(tempfile.mkdtemp()) / "cache.db"
    c = SemanticCache(db_path=str(db), threshold=0.95, enabled=True)
    c.put(_vec(seed=1), "问题A", "答案A", "[]")
    assert c.get(_vec(seed=2), "问题B") is None, "不相似向量不应命中"
    c.close()


def test_disabled_cache_no_op():
    db = Path(tempfile.mkdtemp()) / "cache.db"
    c = SemanticCache(db_path=str(db), enabled=False)
    c.put(_vec(seed=1), "问题", "答案", "[]")
    assert c.get(_vec(seed=1), "问题") is None, "禁用时不应命中"
    assert c.stats() == {"enabled": False, "entries": 0, "total_hits": 0}
    c.close()


def test_ttl_expiry():
    db = Path(tempfile.mkdtemp()) / "cache.db"
    c = SemanticCache(db_path=str(db), threshold=0.9, ttl=1, enabled=True)
    v = _vec(seed=1)
    c.put(v, "问题", "答案", "[]")
    assert c.get(v, "问题") is not None, "TTL 内应命中"
    time.sleep(1.2)
    assert c.get(v, "问题") is None, "TTL 过期后不应命中"
    c.close()


def test_clear():
    db = Path(tempfile.mkdtemp()) / "cache.db"
    c = SemanticCache(db_path=str(db), threshold=0.9, enabled=True)
    c.put(_vec(seed=1), "问题", "答案", "[]")
    assert c.stats()["entries"] == 1
    c.clear()
    assert c.stats()["entries"] == 0
    c.close()