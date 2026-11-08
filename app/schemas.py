"""请求/响应模型（非表模型，纯 Pydantic）。入参/出参分离。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    title: str
    content: str
    source: str = "manual"


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class ChatCitation(BaseModel):
    document_id: int
    title: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[ChatCitation]
    latency_ms: int
    model: str


class MetricsResponse(BaseModel):
    requests: int
    chat_requests: int
    avg_latency_ms: float
    total_tokens: int


# ---------- 纯检索（2026-11-03 新增） ----------
# 动机：一线场景绝大多数问题是「哪份文件写了这条」，不需要 LLM 生成。
# /chat 每次都要调模型，慢且花钱；/search 只返回命中原文，可由调用方直接展示。
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索词或自然语言问题")
    top_k: int = Field(5, ge=1, le=20, description="返回条数上限")
    hybrid: bool = Field(True, description="是否启用 BM25 + 向量 RRF 混合检索")


class SearchHit(BaseModel):
    rank: int          # 1-based 排名，便于前端直接渲染
    score: float       # 融合分（hybrid=True 时为 RRF 累加分）
    document_id: int
    title: str         # 文档标题，即「出处」
    seq: int           # 该 chunk 在文档内的序号，用于定位段落
    text: str          # chunk 原文，不做截断：查出处场景需要完整段落


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHit]
    latency_ms: int


class EvalResult(BaseModel):
    question: str
    expected_doc: str
    retrieved_top1_doc: str
    retrieval_hit: bool
    faithfulness_hit: bool
    answer: str
    # 命中层级透明化（2026-11-03）：区分「聚合页命中」与「原始文档命中」，
    # 避免只看合计命中率时误以为文档层检索正常（当时聚合页兜底掩盖了关键词路失效）。
    wiki_hit: bool = False
    doc_hit: bool = False


class EvalReport(BaseModel):
    total: int
    retrieval_hit_rate: float
    faithfulness_rate: float
    # 命中层级拆分（2026-11-03）：doc=原始文档层，wiki=聚合页层；合计为 retrieval_hit_rate
    doc_hit_rate: float = 0.0
    wiki_hit_rate: float = 0.0
    note: str
    details: List[EvalResult]
