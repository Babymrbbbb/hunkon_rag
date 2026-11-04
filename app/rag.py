"""RAG 管线编排：ingest（灌库）与 chat（问答）。"""
import json
import threading
import time
from typing import List, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.embedder import Embedder, get_embedder
from app.llm import LLMProvider, get_llm_provider
from app.metrics import metrics
from app.models import Chunk, Document, QA
from app.retriever import Retriever, chunk_text, get_doc_titles
from app.schemas import ChatCitation
from app.wiki import WikiPage, build_wiki

SYSTEM_PROMPT = (
    "你是「天鹅公主 SWAN NOBLE」品牌的 AI 知识助手。"
    "只能依据下面提供的资料回答用户问题，不得编造资料之外的信息；"
    "若资料不足以回答，请如实说明，并列出你参考的资料来源。"
)


# ---------- 异步批量审计落库（2026-10-30 压测优化） ----------
# 并发压测时每次 /chat 都同步 commit QA 审计记录，会让所有请求争抢 SQLite 写锁，
# 严重抬高 P95/P99。这里改为：请求只把 QA 放进内存队列，由后台线程批量写库。
# 优点：/chat 不再等待 commit；批量写减少事务次数，大幅降低写锁竞争。
# 代价：极端崩溃场景可能丢失最近若干条审计（可接受，审计非业务关键路径）。
_AUDIT_QUEUE: list = []
_AUDIT_LOCK = threading.Lock()
_AUDIT_STOP = threading.Event()
_AUDIT_THREAD = None


def _audit_worker() -> None:
    """后台线程：批量把队列中的 QA 写入数据库。"""
    from app.database import engine

    while not _AUDIT_STOP.is_set():
        # 取走当前队列全部（批量）
        with _AUDIT_LOCK:
            batch = _AUDIT_QUEUE[:]
            _AUDIT_QUEUE.clear()
        if batch:
            try:
                with Session(engine) as s:
                    for item in batch:
                        s.add(QA(**item))
                    s.commit()
            except Exception:
                # 审计失败不阻断业务；丢弃这批（避免死循环重试）
                pass
        # 无任务时短暂休眠，避免忙等
        _AUDIT_STOP.wait(0.05)


def _ensure_audit_thread() -> None:
    """确保后台审计线程已启动（幂等）。"""
    global _AUDIT_THREAD
    if _AUDIT_THREAD is None or not _AUDIT_THREAD.is_alive():
        _AUDIT_THREAD = threading.Thread(target=_audit_worker, name="qa-audit-writer", daemon=True)
        _AUDIT_THREAD.start()


def _enqueue_audit(question: str, answer: str, citations: str) -> None:
    """把一条 QA 审计放入内存队列（非阻塞）。"""
    _ensure_audit_thread()
    with _AUDIT_LOCK:
        _AUDIT_QUEUE.append(
            {"question": question, "answer": answer, "citations": citations}
        )


def _wiki_context(wiki_pages: List[WikiPage]) -> Tuple[str, List[ChatCitation]]:
    """把聚合页转成上下文与引用（来源列表）。"""
    blocks = []
    citations = []
    for i, page in enumerate(wiki_pages, start=1):
        blocks.append(
            f"[资料{i}]（来自《{page.title}》聚合页）\n"
            f"摘要：{page.summary}\n"
            f"关键事实：{'；'.join(page.facts)}\n"
            f"来源文档：{'、'.join(page.sources)}"
        )
        citations.append(
            ChatCitation(
                document_id=-1,  # 聚合页无 document_id
                title=page.title,
                snippet=page.summary[:120],
            )
        )
    return "\n\n".join(blocks), citations


def ingest(
    session: Session,
    title: str,
    content: str,
    source: str = "manual",
    embedder: Embedder = None,
) -> Document:
    embedder = embedder or get_embedder()
    doc = Document(title=title, source=source, content=content)
    session.add(doc)
    session.commit()
    session.refresh(doc)

    chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    for seq, ch in enumerate(chunks):
        vec = embedder.embed(ch)
        session.add(
            Chunk(document_id=doc.id, seq=seq, text=ch, embedding=json.dumps(vec))
        )
    session.commit()
    return doc


def chat(
    session: Session,
    question: str,
    top_k: int = None,
    llm: LLMProvider = None,
    embedder: Embedder = None,
    retriever: Retriever = None,
    use_wiki: bool = True,
    role: str = "",
) -> Tuple[str, List[ChatCitation], int, int]:
    llm = llm or get_llm_provider()
    embedder = embedder or get_embedder()
    retriever = retriever or Retriever(embedder, settings.top_k)

    t0 = time.time()

    # 聚合层优先：命中聚合页则用聚合上下文（raw→wiki 双轨的 wiki 层）
    wiki_hits: List[WikiPage] = []
    if use_wiki:
        wiki = build_wiki(save=False)  # save=False：索引变化时构建内部才落盘，避免并发写
        wiki_hits = wiki.search(question)

    # RAG 常规路径：混合检索召回原始 chunk
    # 查询改写（2026-11-04 接入，默认关闭）：口语化问法先经 LLM 改写为检索友好形式。
    # 默认 RAG_QUERY_REWRITE=0 时原样返回，零开销。
    from app.rewrite import rewrite_query

    search_question, _changed = rewrite_query(question)
    qvec = embedder.embed_query(search_question)
    hits = retriever.search(session, qvec, top_k, query_text=search_question, use_hybrid=True)

    doc_titles = get_doc_titles(session)

    # 多租户 RBAC（2026-11-04 新增）：按角色过滤不可见文档的 chunk。
    # 仅当显式传入 role 且为有效角色时生效；否则不过滤（兼容旧调用）。
    if role:
        from app.rbac import filter_hits_by_role, get_doc_sources

        doc_sources = get_doc_sources(session)
        hits = filter_hits_by_role(hits, role, doc_sources)

    # CRAG 自纠错（2026-11-04 新增，默认关闭）：LLM 判断上下文相关性，
    # 不足则改写查询重试一次。仅 RAG_CRAG=1 且真实 LLM 时生效，否则原样返回。
    from app.crag import corrective_rerank

    hits, _used_rewrite, _retried = corrective_rerank(
        question,
        hits,
        session,
        embedder,
        retriever,
        top_k,
        search_question,
        doc_titles,
    )

    chunk_context = "\n\n".join(
        f"[资料{i+1}]（来自《{doc_titles.get(chunk.document_id, '未知')}》）\n{chunk.text}"
        for i, (_score, chunk) in enumerate(hits)
    )

    # 上下文 = 聚合页（若有优先）+ 原始 chunk（补充细节）
    contexts = []
    citations: List[ChatCitation] = []
    if wiki_hits:
        wiki_ctx, wiki_cits = _wiki_context(wiki_hits)
        contexts.append(wiki_ctx)
        citations.extend(wiki_cits)
    if chunk_context.strip():
        contexts.append(chunk_context)
        for i, (_score, chunk) in enumerate(hits):
            citations.append(
                ChatCitation(
                    document_id=chunk.document_id,
                    title=doc_titles.get(chunk.document_id, "未知"),
                    snippet=chunk.text[:120],
                )
            )

    context = "\n\n".join(contexts)
    user_msg = f"用户问题：{question}\n\n可参考资料：\n{context}"

    # 语义缓存（2026-11-04 接入）：语义等价的重复问题直接返回历史答案，
    # 跳过 LLM 调用，节省成本与延迟。命中时仍返回本次检索的 citations 作为引用，
    # 保证回答可溯源（与生产库语义缓存语义一致）。
    from app.cache import SemanticCache

    _cache = SemanticCache()
    cached = _cache.get(qvec, question)
    if cached is not None:
        answer, cached_citations, _hit_count, _cached_q = cached
        latency = int((time.time() - t0) * 1000)
        approx_tokens = (len(answer) + len(user_msg)) // 4
        metrics.record_request()
        metrics.record_chat(latency, approx_tokens)
        _enqueue_audit(
            question=question,
            answer=answer,
            citations=json.dumps([c.model_dump() for c in citations], ensure_ascii=False),
        )
        return answer, citations, latency, approx_tokens

    answer = llm.complete(SYSTEM_PROMPT, user_msg)

    # 写入语义缓存（仅真实 LLM 回答才缓存，避免缓存 mock 占位回答）
    if settings.llm_provider == "real":
        try:
            _cache.put(
                qvec,
                question,
                answer,
                json.dumps([c.model_dump() for c in citations], ensure_ascii=False),
            )
        except Exception:
            pass

    latency = int((time.time() - t0) * 1000)
    # token 近似（mock 与真实通用）：字符数 / 4
    approx_tokens = (len(answer) + len(user_msg)) // 4
    metrics.record_request()
    metrics.record_chat(latency, approx_tokens)

    # 落库审计（异步批量：不阻塞 /chat 响应，避免并发写锁竞争）
    _enqueue_audit(
        question=question,
        answer=answer,
        citations=json.dumps([c.model_dump() for c in citations], ensure_ascii=False),
    )

    return answer, citations, latency, approx_tokens
