"""请求/响应模型（非表模型，纯 Pydantic）。入参/出参分离。"""
from typing import List, Optional

from pydantic import BaseModel


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
    top1_source: str = "none"  # wiki | doc | none


class EvalReport(BaseModel):
    total: int
    retrieval_hit_rate: float
    faithfulness_rate: float
    # 命中层级拆分（2026-11-03）：doc=原始文档层，wiki=聚合页层；合计为 retrieval_hit_rate
    doc_hit_rate: float = 0.0
    wiki_hit_rate: float = 0.0
    note: str
    details: List[EvalResult]
