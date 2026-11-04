"""查询改写——生产库 query_rewrite.py 的开源复刻。

能力等价：
  - 口语化/指代不明/缺实体的问法先经 LLM 改写为检索友好形式
  - 改写失败/限流时静默降级为原始查询，绝不阻塞主链路
  - 默认关闭（RAG_QUERY_REWRITE=1 开启），聚合层语义兜底已覆盖多数口语化问法
  - 改写结果缓存（SQLite，7 天），避免重复调用 LLM
  - 质量校验：改写结果必须保留原文关键实词

与生产库差异：
  - 生产库用 rag_qa.llm_complete（urllib 直连 + 环境变量端点）
  - hunkon_rag 用 app.llm.get_llm_provider()（LLMProvider 抽象，配置化）
  - 生产库环境变量 KB_QUERY_REWRITE；hunkon_rag 用 RAG_QUERY_REWRITE

用法：
  from app.rewrite import rewrite_query
  rewritten, changed = rewrite_query("那家卖坚果礼盒的怎么样？")
"""
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "rewrite_cache.db"

_REWRITE_SYSTEM = (
    "你是检索查询改写器。用户的问题可能口语化、指代不明、缺实体。"
    "你的任务是【最小改写】：在保留原问题全部关键实词和查询意图的前提下，"
    "只做两件事：① 把指代词（他们/那家/这家/这个）补全为明确的品牌或对象名；"
    "② 去掉口头语气词。改写结果必须：\n"
    "1. 仍是一个查询问题，不是澄清/反问句，不得出现『请提供』『您指的是』『以便』等字眼；\n"
    "2. 保留原问题中的数字、品牌名、关键词（如『多少家』『门店』必须保留）；\n"
    "3. 与原文同义，不新增不存在的限定；\n"
    "4. 输出一行改写结果，不要解释、不要引号、不要前缀。\n"
    "若原始问题本身已规范、无需改写，原样输出。"
)


class QueryRewriteCache:
    """改写结果缓存，避免重复调用 LLM。"""

    def __init__(self, db_path=None, ttl=7 * 24 * 3600):
        self.db_path = str(db_path) if db_path else str(DEFAULT_DB)
        self.ttl = ttl
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rewrite_cache (
                question_hash TEXT PRIMARY KEY,
                original TEXT NOT NULL,
                rewritten TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, question):
        h = hashlib.md5(question.encode("utf-8")).hexdigest()
        try:
            row = self._conn.execute(
                "SELECT rewritten FROM rewrite_cache WHERE question_hash = ? AND created_at > ?",
                (h, time.time() - self.ttl),
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def put(self, question, rewritten):
        h = hashlib.md5(question.encode("utf-8")).hexdigest()
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO rewrite_cache (question_hash, original, rewritten, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (h, question, rewritten, time.time()),
            )
            self._conn.commit()
        except Exception:
            pass

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def _enabled() -> bool:
    """默认关闭（与生产库一致）：聚合层语义兜底已覆盖口语化问法。
    开启：RAG_QUERY_REWRITE=1。"""
    return os.environ.get("RAG_QUERY_REWRITE", "0").lower() in (
        "1", "true", "yes", "on")


# 高频虚词（用于核心词提取，避免把语气词当关键实词）
_STOPWORDS = set(
    "的是在有着与和以为其也将并而之及等个们都可所被把要对中内上下面后前这那"
    "或就也很更最再没无此从向当该各它她你我又但因若由、，。：；！？（）《》"
    "「」*_~#@%&$=+-/\\| "
    "的了是在与和及等你我他她它这那有就都为又再但而或之其此如于对从到被把让因为所以如果"
    "一个没有我们你们他们可以这些那些怎么什么自己已经还是不是时候进行以及"
    "请问您指的是哪个品牌或店铺请提供具体名称以便查询"
)


def _keeps_core_terms(original: str, rewritten: str) -> bool:
    """质量校验：改写结果必须保留原文的关键实词。
    用 2-3 字滑动窗口提取原文关键词，检查改写结果至少包含一个关键窗口词。
    """
    # 1. 数字必须保留
    nums = re.findall(r"\d+(?:\.\d+)?", original)
    for n in nums:
        if n in rewritten:
            return True
    # 2. 品牌名必须保留
    for brand in ("Hunkon", "玛芝莲", "金粒门"):
        if brand in original and brand in rewritten:
            return True
    # 3. 滑动窗口提取 2-3 字关键词，至少一个保留
    text = re.sub(r"[^\u4e00-\u9fff]", "", original)  # 只留汉字
    windows = set()
    for size in (3, 2):
        for i in range(len(text) - size + 1):
            w = text[i:i + size]
            if w not in _STOPWORDS:
                windows.add(w)
    for w in windows:
        if w in rewritten:
            return True
    return False


def _looks_rewritable(question: str) -> bool:
    """判断是否需要改写：含口语词、指代词、缺实体特征时返回 True。"""
    colloquial = (
        "多少家", "几间", "几家", "多少个", "有没有", "是不是", "咋", "怎么", "为啥",
        "他们", "她们", "它们", "那家", "这家", "那个", "这个", "哪里", "哪儿",
        "总共", "一共", "到底", "来着", "吧", "呢", "呀", "嘛",
    )
    for w in colloquial:
        if w in question:
            return True
    # 无品牌词且较短（< 12 字符）的裸问法，大概率缺实体
    if len(question) < 12 and "Hunkon" not in question and "玛芝莲" not in question:
        return True
    return False


def rewrite_query(question: str):
    """改写查询。返回 (改写后的查询, 是否发生改写)。
    任何异常/限流/改写失败都降级为原始查询。
    """
    if not _enabled():
        return question, False
    if not _looks_rewritable(question):
        return question, False

    cache = QueryRewriteCache()
    try:
        cached = cache.get(question)
        if cached:
            # 缓存命中也要过质量校验（防止旧版坏改写残留）
            if cached == question or not _keeps_core_terms(question, cached):
                cache.put(question, question)
                return question, False
            return cached, True

        from app.llm import get_llm_provider

        llm = get_llm_provider()
        rewritten = llm.complete(_REWRITE_SYSTEM, question)
        rewritten = (rewritten or "").strip().strip('"').strip("'").strip()
        # 改写结果异常时降级
        if not rewritten or rewritten.startswith(("HTTP", "429", "无 API key")):
            cache.put(question, question)
            return question, False
        if rewritten == question:
            cache.put(question, question)
            return question, False
        if len(rewritten) > 200:
            cache.put(question, question)
            return question, False
        # 质量校验：必须保留原文关键实词
        if not _keeps_core_terms(question, rewritten):
            cache.put(question, question)
            return question, False
        cache.put(question, rewritten)
        return rewritten, True
    except Exception:
        return question, False
    finally:
        try:
            cache.close()
        except Exception:
            pass