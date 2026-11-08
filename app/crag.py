"""CRAG 自纠错模块（Corrective RAG）——参考生产级 RAG 的纠错范式实现。

设计：
  1. 检索后，用 LLM 判断「当前上下文是否足以回答问题」
     - relevant：上下文充足，直接用
     - irrelevant：上下文不相关 → 改写查询重试一次
     - ambiguous：上下文模糊 → 改写查询重试一次
  2. 重试用「更简单直白」的改写（去掉修饰词、保留核心实体），重新检索
  3. 如果重试后的上下文仍不足，则保留第一次的（避免空上下文）

原则：
  - 默认关闭（RAG_CRAG=1 开启），因为 mock LLM 的判断无意义
  - 任何失败（LLM 异常/解析失败）静默降级为「使用原始检索结果」，绝不阻断主链路
  - 只在真实 LLM 模式下有意义，mock 模式下开关无效（判断不可靠）

用法：
  from app.crag import corrective_rerank
  hits, used_rewrite, retried = corrective_rerank(
      question, hits, session, embedder, retriever, top_k, search_question, doc_titles)
"""
from typing import List, Tuple

from app.config import settings

_GRADE_SYSTEM = (
    "你是检索质量评估器。给定用户问题和检索到的参考资料，判断参考资料"
    "是否足以回答用户问题。只输出一个词：\n"
    "relevant（资料充分相关，足以回答）\n"
    "irrelevant（资料与问题无关，或缺少关键信息，不足以回答）\n"
    "ambiguous（资料部分相关，但关键信息缺失或模糊）\n"
    "不要输出任何其他内容。"
)

_REWRITE_SYSTEM = (
    "你是检索查询改写器。用户的问题第一次检索没有找到足够相关的资料，"
    "请你把问题改写为【更简单直白】的检索查询：\n"
    "1. 去掉修饰词、口语语气词、不必要的从句；\n"
    "2. 保留核心实体名（品牌名、产品名、城市名）和关键数字；\n"
    "3. 如果原文有指代词（他们/那家/这个），补全为具体对象名；\n"
    "4. 输出一行改写结果，不要解释、不要引号。"
)


def _grade(question: str, context: str) -> str:
    """用 LLM 判断上下文相关性。返回 'relevant' / 'irrelevant' / 'ambiguous'。"""
    from app.llm import get_llm_provider

    llm = get_llm_provider()
    user = f"用户问题：{question}\n\n参考资料：\n{context[:3000]}"
    try:
        raw = llm.complete(_GRADE_SYSTEM, user)
    except Exception:
        return "relevant"  # 失败静默降级：视为相关，不打断
    raw = (raw or "").strip().lower()
    for word in ("irrelevant", "ambiguous"):
        if word in raw:
            return word
    return "relevant"


def _rewrite(question: str) -> str:
    """用 LLM 改写查询为更简单直白的形式。失败返回原文。"""
    from app.llm import get_llm_provider

    llm = get_llm_provider()
    try:
        raw = llm.complete(_REWRITE_SYSTEM, question)
    except Exception:
        return question
    rewritten = (raw or "").strip().strip('"').strip("'").strip()
    if not rewritten or rewritten.startswith(("HTTP", "429", "无 API key")):
        return question
    if len(rewritten) > 200:
        return question
    return rewritten or question


def _hits_to_text(hits, doc_titles) -> str:
    """把检索结果转成文本（供相关性判断）。"""
    if not hits:
        return ""
    blocks = []
    for _score, chunk in hits:
        title = doc_titles.get(chunk.document_id, "未知")
        blocks.append(f"《{title}》：{chunk.text[:200]}")
    return "\n\n".join(blocks)


def corrective_rerank(
    question: str,
    hits,
    session,
    embedder,
    retriever,
    top_k: int,
    search_question: str,
    doc_titles,
) -> Tuple[List, bool, bool]:
    """CRAG 主入口（基于已有检索结果做纠错）。

    参数：
    - hits：第一次检索的结果（RRF 后）
    - 其余同 retriever.search 的上下文
    返回 (最终 hits, 是否改写, 是否重试)。
    """
    if not settings.crag:
        return hits, False, False
    # mock LLM 模式下判断不可靠，直接返回
    if settings.llm_provider != "real":
        return hits, False, False

    context_text = _hits_to_text(hits, doc_titles)
    if not context_text.strip():
        return hits, False, False

    verdict = _grade(question, context_text)
    if verdict == "relevant":
        return hits, False, False

    # 改写并重试一次
    rewritten = _rewrite(search_question)
    if not rewritten or rewritten == search_question:
        return hits, False, False  # 改写失败，保留原结果

    try:
        qvec = embedder.embed_query(rewritten)
        new_hits = retriever.search(session, qvec, top_k, query_text=rewritten, use_hybrid=True)
    except Exception:
        return hits, False, False

    # 重试后若仍有结果，用新结果；否则保留原结果
    if new_hits:
        return new_hits, True, True
    return hits, False, False