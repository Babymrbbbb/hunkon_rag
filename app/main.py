"""FastAPI 入口：路由层 + Depends 注入 + 全局鉴权 + 统一错误信封 + lifespan。"""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status as http_status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from sqlmodel import Session

from app.config import settings
from app.database import engine, get_session, init_db
from app.embedder import get_embedder
from app.eval import run_eval
from app.feedback import FeedbackStore
from app.llm import get_llm_provider
from app.rbac import filter_hits_by_role, get_doc_sources, role_for_key
from app.metrics import metrics
from app.rag import chat as rag_chat, ingest
from app.retriever import Retriever, chunk_text, get_doc_titles
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    MetricsResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.seed import reseed_if_needed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 并发压测优化（2026-10-30）：FastAPI 同步端点由 anyio 线程池执行，
    # 默认线程池上限 40。压测并发 100 时 60 个请求排队，P50 被队列时间抬高。
    # 这里把 anyio 线程池调大，让同步端点能同时处理更多请求。
    try:
        import anyio
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 128
    except Exception:
        pass  # 旧版本 anyio 无此 API 则忽略

    # 启动：建表 + 索引自愈合（源文件改动时自动重建，防止「改源忘灌库」）
    init_db()
    if settings.auto_seed:
        with Session(engine) as s:
            reseed_if_needed(s)
    yield


async def get_api_key(request: Request, x_api_key: str = Header(None, alias="X-API-Key")):
    """全局依赖：除演示页与文档外，所有接口强制带 X-API-Key，缺失/错误返回 401 统一信封。

    白名单（免鉴权）是为了让访问者打开页面即可体验，不破坏「全局鉴权」的架构叙事：
    /、/docs、/openapi.json、/redoc 均可直接访问；业务接口 /chat 等仍强制 X-API-Key。

    2026-11-04 多租户：key 可以是 staff/franchise/ops 三档之一（见 rbac.role_for_key）。
    """
    if request.url.path in ("/", "/docs", "/openapi.json", "/redoc", "/monitor", "/reports/history.json"):
        return None
    if not x_api_key or not role_for_key(x_api_key):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="无效或缺失 API Key"
        )
    return x_api_key


async def get_current_role(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """返回当前请求的角色（staff/franchise/ops）。供 /chat 与 /search 做数据过滤。"""
    return role_for_key(x_api_key or "")


app = FastAPI(
    title="演示品牌 RAG 知识问答服务",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(get_api_key)],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "请求体校验失败",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """演示页（深色主题对话 UI）。免鉴权，便于直接体验。"""
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    """LLMOps 监控看板（2026-10-30 新增，突破1）。免鉴权，展示评测历史 + 运行时指标。"""
    html_path = Path(__file__).resolve().parent / "static" / "monitor.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/reports/history.json")
def report_history():
    """评测历史（供监控看板读取）。免鉴权。"""
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    history_path = reports_dir / "history.json"
    if not history_path.exists():
        return []
    return json.loads(history_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": settings.llm_provider,
        "embedder": settings.embedder_provider,
        "model": settings.llm_model,
    }


@app.post("/ingest")
def ingest_doc(req: IngestRequest, session: Session = Depends(get_session)):
    embedder = get_embedder()
    doc = ingest(session, req.title, req.content, req.source, embedder=embedder)
    n_chunks = len(chunk_text(req.content, settings.chunk_size, settings.chunk_overlap))
    return {"id": doc.id, "title": doc.title, "chunks": n_chunks}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    req: ChatRequest,
    session: Session = Depends(get_session),
    role: str = Depends(get_current_role),
):
    llm = get_llm_provider()
    embedder = get_embedder()
    retriever = Retriever(embedder, settings.top_k)
    answer, citations, latency, _tokens = rag_chat(
        session, req.question, top_k=req.top_k, llm=llm, embedder=embedder, retriever=retriever,
        role=role,
    )
    return ChatResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency,
        model=f"{settings.llm_provider}:{settings.llm_model}",
    )


@app.post("/search", response_model=SearchResponse)
def search_endpoint(
    req: SearchRequest,
    session: Session = Depends(get_session),
    role: str = Depends(get_current_role),
):
    """纯检索：只返回命中片段与出处，不调用 LLM（2026-11-03 新增）。

    与 /chat 的分工：
    - /search：秒级、零 LLM 成本，答「哪份文件写了这条规定」。一线高频场景。
    - /chat：慢、消耗模型额度，答「所以到底该怎么办」。适合复杂追问。
    返回 text 为完整 chunk 原文，便于直接作为「出处」展示，不做摘要截断。

    2026-11-04 多租户：按角色过滤不可见文档的 chunk。
    """
    t0 = time.time()
    embedder = get_embedder()
    retriever = Retriever(embedder, settings.top_k)
    # 查询改写（2026-11-04 接入，默认关闭）：口语化问法先经 LLM 改写为检索友好形式。
    from app.rewrite import rewrite_query

    search_query, _changed = rewrite_query(req.query)
    hits = retriever.search(
        session,
        embedder.embed_query(search_query),
        req.top_k,
        query_text=search_query,
        use_hybrid=req.hybrid,
    )
    doc_titles = get_doc_titles(session)
    # 多租户过滤：按当前角色剔除不可见文档
    doc_sources = get_doc_sources(session)
    hits = filter_hits_by_role(hits, role, doc_sources)
    payload = [
        SearchHit(
            rank=i,
            score=round(float(score), 6),
            document_id=chunk.document_id,
            title=doc_titles.get(chunk.document_id, "未知"),
            seq=chunk.seq,
            text=chunk.text,
        )
        for i, (score, chunk) in enumerate(hits, start=1)
    ]
    metrics.record_request()
    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        hits=payload,
        latency_ms=int((time.time() - t0) * 1000),
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return MetricsResponse(
        requests=metrics.requests,
        chat_requests=metrics.chat_requests,
        avg_latency_ms=round(metrics.avg_latency, 2),
        total_tokens=metrics.total_tokens,
    )


@app.post("/eval")
def eval_endpoint(session: Session = Depends(get_session)):
    return run_eval(session)


@app.post("/api/feedback")
def add_feedback(req: dict, session: Session = Depends(get_session)):
    """记录用户对问答结果的反馈（点赞/点踩）。

    请求体：{question, answer, citations?, rating(1/-1), reason?}
    """
    store = FeedbackStore()
    fid = store.add(
        question=req.get("question", ""),
        answer=req.get("answer", ""),
        citations=req.get("citations") or [],
        rating=req.get("rating", 1),
        reason=req.get("reason", ""),
    )
    store.close()
    if fid is None:
        raise HTTPException(status_code=500, detail="反馈写入失败")
    return {"id": fid, "ok": True}


@app.get("/api/feedback/stats")
def feedback_stats():
    """反馈统计：总数 / 点赞 / 点踩。"""
    store = FeedbackStore()
    try:
        return store.stats()
    finally:
        store.close()


@app.get("/api/feedback/list")
def feedback_list(limit: int = 50, rating: int = None):
    """查询反馈记录（可按 rating 过滤）。"""
    store = FeedbackStore()
    try:
        return store.list(limit=limit, rating=rating)
    finally:
        store.close()
