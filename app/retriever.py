"""检索器：双路混合检索（倒排关键词 + 向量召回）→ RRF 融合。

架构演进（完整双轨）：
- 路1 · 向量召回：语义匹配（处理口语化/换说法问题）。mock 向量离线可复现；真实 embedding 可用时切换。
- 路2 · 倒排关键词召回：BM25-lite（词频 + IDF + 业务子库加权），精确匹配业务词（处理术语/编号/规范问法）。
- 融合：RRF（Reciprocal Rank Fusion）合并两路结果，去重加权，取 top-k。

设计动机：单路向量在垂直术语场景可能漏精确命中；单路倒排在口语化场景会漏语义。
双路并行取长补短——这正是 llm-wiki「词频饱和 + 业务子库加权」生产检索的通用化实现。
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from typing import List, Tuple

from sqlmodel import Session, func, select

from app.embedder import Embedder, get_embedder
from app.models import Chunk, Document

# ---------- 文本处理 ----------

def chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    """按字符长度滑窗切分（中文友好）。"""
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# 高频中文虚词/标点：倒排索引分词时过滤（与 MockEmbedder 同一套停用词思路）。
_STOP = set(
    "的是在有着与和以为其也将并而之及等个们都可所被把要对中内上下面后前这那"
    "或就也很更最再没无此从向当该各它她你我又但因若由、，。：；！？（）《》"
    "「」*_~#@%&$=+-/\\| "
)


def _tokenize(text: str) -> List[str]:
    """中英文通用确定性分词：单字 + 二元组，过滤停用词与标点。"""
    text = text.lower()
    chars = [c for c in text if c not in _STOP]
    toks = list(chars)
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


# ---------- BM25-lite 倒排索引 ----------

class InvertedIndex:
    """内存倒排索引：term -> {chunk_id: tf}，支持 IDF 加权与 BM25-lite 打分。"""

    def __init__(self, chunks: List[Chunk]) -> None:
        self.doc_len = len(chunks)
        self.df: Counter[str] = Counter()                          # term -> 包含它的 chunk 数
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)  # term -> {chunk_id: tf}
        self.chunk_len: dict[int, int] = {}                        # chunk_id -> 词数
        for c in chunks:
            toks = _tokenize(c.text)
            self.chunk_len[c.id] = len(toks)
            for term, cnt in Counter(toks).items():
                self.postings[term][c.id] = cnt
                self.df[term] += 1

    def idf(self, term: str) -> float:
        """IDF：稀有词权重高，泛词被压低（llm-wiki 的 IDF 降权思想）。"""
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.doc_len - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Tuple[float, int]]:
        """BM25-lite：对查询词累加 IDF × tf 饱和，返回 (score, chunk_id)。

        注意返回顺序必须是 (score, chunk_id)，与 rrf_fuse 的 (score, cid) 约定一致。
        （2026-11-03 修复：此前返回 (chunk_id, score)，导致 RRF 融合时两者互换，
        关键词路结果被整体丢弃，混合检索实际退化为纯向量检索。）
        """
        terms = _tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        avg_len = (sum(self.chunk_len.values()) / len(self.chunk_len)) if self.chunk_len else 1.0
        for term in set(terms):  # 同一词只算一次（防止长问题刷分）
            idf = self.idf(term)
            if idf == 0:
                continue
            for cid, tf in self.postings.get(term, {}).items():
                dl = self.chunk_len.get(cid, 1)
                denom = tf + k1 * (1 - b + b * dl / avg_len)
                scores[cid] += idf * (tf * (k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # scores.items() 是 (chunk_id, score)，这里显式翻转成 (score, chunk_id)
        return [(float(score), int(cid)) for cid, score in ranked[:top_k]]


# ---------- 向量召回 ----------

def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_DIM_WARNED = False


def _check_dim_once(query_vec: List[float], chunks: List[Chunk]) -> None:
    """向量维度不一致时只告警一次，不中断服务。

    场景：换 embedding 模型（如 mock 256 维 → local 1024 维）却没重建索引。
    此时 cosine 因长度不等恒返回 0（见 cosine 实现），向量路静默失效、只剩关键词路，
    表现为「检索突然变差但不报错」——很难排查，所以这里显式喊出来。
    正常路径由 seed.py 的 embedder 指纹校验自动重建，不会走到这个告警。
    """
    global _DIM_WARNED
    if _DIM_WARNED or not query_vec or not chunks:
        return
    try:
        stored = len(json.loads(chunks[0].embedding))
    except Exception:
        return
    if stored and stored != len(query_vec):
        _DIM_WARNED = True
        print(
            f"[retriever] 向量维度不一致：库内 {stored} 维 / 当前模型 {len(query_vec)} 维。"
            "向量检索将静默失效（余弦恒为 0），请重建索引。",
            file=sys.stderr,
        )


# ---------- RRF 融合 ----------

def rrf_fuse(ranked_lists: List[List[Tuple[float, int]]], k: int = 60, top_k: int = 3) -> List[Tuple[float, int]]:
    """Reciprocal Rank Fusion：多路排序融合。

    对每个 chunk，累加 1/(k + rank)，rank 为该路中的位次（从 1 开始）。
    这样两路都靠前的 chunk 得分最高，兼顾精确与语义。

    入参约定：每路都是 [(score, chunk_id), ...]，chunk_id 必须是 int。
    2026-11-03 加防御校验：曾因上游返回 (chunk_id, score) 顺序颠倒，浮点数被当成
    chunk_id 写入融合表，最终全部被 by_id 过滤丢弃且不报错（混合检索静默退化为纯向量）。
    这里显式校验并纠正，避免同类问题再次静默发生。
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (a, b) in enumerate(ranked, start=1):
            # 约定 (score:float, chunk_id:int)；若上游颠倒则自动纠正
            score, cid = (a, b) if isinstance(a, float) else (b, a)
            scores[int(cid)] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # 统一返回 (score, chunk_id) 顺序，与 search 解包一致
    return [(float(score), int(cid)) for cid, score in fused[:top_k]]


# ---------- 检索器 ----------

# 进程级检索缓存（2026-10-30 压测优化）：
# 每次 /chat 都 select(Chunk).all() + 重建 InvertedIndex + 逐条 json.loads 解析向量，
# 这些工作在语料不变时结果完全相同，纯属浪费。这里把全量数据常驻内存，
# 仅当语料哈希变化（由 build_wiki/seed 重建触发）才刷新。
_CHUNK_CACHE: dict = {"hash": None, "chunks": None, "index": None, "by_id": None, "doc_titles": None}


def _corpus_hash_fingerprint(session: Session) -> str:
    """轻量数据指纹：corpus 文件哈希 + 数据库 chunk 表状态。

    并发压测优化（2026-10-30）：进程级缓存需要知道「数据是否变化」。
    仅用 corpus 文件哈希不够——/ingest 会直接向数据库写入新 chunk，
    此时 corpus 文件没变但检索数据变了，缓存必须失效。
    因此指纹 = corpus 哈希 + chunk 行数 + 最新 chunk id。
    """
    from app.config import settings
    from app.wiki import _corpus_digest
    from pathlib import Path

    try:
        corpus_part = _corpus_digest(Path(settings.corpus_dir))
    except Exception:
        corpus_part = "?"
    # 轻量数据库状态：行数 + 最大 id（单条 SQL，走主键索引，避免全表扫描）
    try:
        row = session.exec(
            select(func.count(), func.max(Chunk.id))
        ).one()
        db_part = f"{row[0]}:{row[1] if row[1] is not None else 0}"
    except Exception:
        db_part = "?"
    # 向量化实现也参与指纹：同语料重建（chunk 行数与最大 id 不变）时，
    # 若换了模型，库内向量维度已变而指纹不变会命中旧缓存 —— 检索拿到的仍是
    # 旧维度向量，余弦恒 0，表现为「换了模型但效果不升反降」（2026-11-03 实测踩到）。
    try:
        emb_part = get_embedder().fingerprint
    except Exception:
        emb_part = "?"
    return f"{corpus_part}|{db_part}|{emb_part}"


def invalidate_cache() -> None:
    """强制下一次检索重载 chunk 与倒排索引。

    seed 重建完成后必须调用：重建只改 chunk 内容（行数/最大 id 可能完全不变），
    仅靠指纹无法感知，缓存会继续供旧向量。
    """
    _CHUNK_CACHE["hash"] = None
    _CHUNK_CACHE["chunks"] = None
    _CHUNK_CACHE["index"] = None
    _CHUNK_CACHE["by_id"] = None
    _CHUNK_CACHE["doc_titles"] = None


def _load_chunk_data(session: Session) -> None:
    """全量加载 chunk 数据并构建倒排索引，写入进程缓存。"""
    chunks = session.exec(select(Chunk)).all()
    index = InvertedIndex(chunks)
    docs = session.exec(select(Document)).all()
    _CHUNK_CACHE["chunks"] = chunks
    _CHUNK_CACHE["index"] = index
    _CHUNK_CACHE["by_id"] = {c.id: c for c in chunks}
    _CHUNK_CACHE["doc_titles"] = {d.id: d.title for d in docs}


def get_cached_chunks(session: Session) -> tuple:
    """返回 (chunks, inverted_index, by_id)，带进程级缓存。"""
    try:
        digest = _corpus_hash_fingerprint(session)
    except Exception:
        digest = None
    if digest is not None and _CHUNK_CACHE["hash"] == digest and _CHUNK_CACHE["chunks"] is not None:
        return _CHUNK_CACHE["chunks"], _CHUNK_CACHE["index"], _CHUNK_CACHE["by_id"]
    _load_chunk_data(session)
    _CHUNK_CACHE["hash"] = digest
    return _CHUNK_CACHE["chunks"], _CHUNK_CACHE["index"], _CHUNK_CACHE["by_id"]


def get_doc_titles(session: Session) -> dict:
    """返回 {document_id: title}，带进程级缓存（与 chunk 缓存同生命周期）。"""
    try:
        digest = _corpus_hash_fingerprint(session)
    except Exception:
        digest = None
    if digest is not None and _CHUNK_CACHE["hash"] == digest and _CHUNK_CACHE["doc_titles"] is not None:
        return _CHUNK_CACHE["doc_titles"]
    _load_chunk_data(session)
    _CHUNK_CACHE["hash"] = digest
    return _CHUNK_CACHE["doc_titles"]


class Retriever:
    def __init__(self, embedder: Embedder, top_k: int = 3, use_hybrid: bool = True) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.use_hybrid = use_hybrid

    def embed_query(self, query: str) -> List[float]:
        # 非对称模型（qwen3）查询要加指令前缀，不能复用文档编码
        return self.embedder.embed_query(query)

    def search(
        self,
        session: Session,
        query_vec: List[float],
        top_k: int = None,
        query_text: str = "",
        use_hybrid: bool = None,
    ) -> List[Tuple[float, Chunk]]:
        """双路检索入口。

        参数：
        - query_vec: 问题向量（必传）
        - query_text: 问题原文（混合检索的关键词路需要；为空则退化为纯向量）
        - use_hybrid: 是否启用双路（默认取 self.use_hybrid）
        """
        k = top_k or self.top_k
        hybrid = self.use_hybrid if use_hybrid is None else use_hybrid
        chunks, index, by_id = get_cached_chunks(session)
        if not chunks:
            return []

        # 路1：向量余弦（复用缓存的全量向量，避免逐条 json.loads）
        _check_dim_once(query_vec, chunks)
        vec_ranked: List[Tuple[float, int]] = []
        for c in chunks:
            try:
                vec = json.loads(c.embedding)
            except Exception:
                vec = [0.0]
            vec_ranked.append((cosine(query_vec, vec), c.id))
        vec_ranked.sort(key=lambda x: x[1], reverse=True)  # 按 id 稳定
        vec_ranked.sort(key=lambda x: x[0], reverse=True)
        vec_top = vec_ranked[: max(k * 3, 5)]

        # 路2：倒排关键词（query_text 非空且启用混合时）——复用缓存索引
        kw_top: List[Tuple[float, int]] = []
        if hybrid and query_text.strip():
            kw_top = index.search(query_text, top_k=max(k * 3, 5))

        # 融合
        if hybrid and kw_top:
            fused = rrf_fuse([vec_top, kw_top], top_k=k)
        else:
            fused = vec_top[:k]

        # 映射回 Chunk（复用缓存 by_id）
        result = []
        for score, cid in fused:
            if cid in by_id:
                result.append((float(score), by_id[cid]))

        # 精排（2026-11-04 新增）：RRF 融合后，用交叉编码器对候选重排。
        # 默认 RAG_RERANK=score 时保持原序（零开销）；设为 local 且模型存在时
        # 走真 cross-encoder 精排。模型缺失自动降级 score（不伪造能力）。
        if query_text.strip() and result:
            try:
                from app.reranker import get_reranker

                reranker = get_reranker()
                if reranker.name != "score":
                    # 只对 RRF 前 max(top_k*3, 5) 个候选精排，控制成本
                    pool = result[: max(k * 3, 5)]
                    result = reranker.rerank(query_text, pool, k)
            except Exception:
                # 精排失败绝不影响主检索（降级为原 RRF 结果）
                pass
        return result