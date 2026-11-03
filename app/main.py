"""FastAPI 入口：路由层 + Depends 注入 + 全局鉴权 + 统一错误信封 + lifespan。"""
import json
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
from app.llm import get_llm_provider
from app.metrics import metrics
from app.rag import chat as rag_chat, ingest
from app.retriever import Retriever, chunk_text
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    MetricsResponse,
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

    白名单（免鉴权）是为了让面试官打开页面即可体验，不破坏「全局鉴权」的架构叙事：
    /、/docs、/openapi.json、/redoc 均可直接访问；业务接口 /chat 等仍强制 X-API-Key。
    """
    if request.url.path in ("/", "/docs", "/openapi.json", "/redoc", "/monitor", "/reports/history.json"):
        return None
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="无效或缺失 API Key"
        )
    return x_api_key


app = FastAPI(
    title="天鹅公主 RAG 知识问答服务",
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
    """演示页（黑金主题对话 UI）。免鉴权，方便面试官直接体验。"""
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
def chat_endpoint(req: ChatRequest, session: Session = Depends(get_session)):
    llm = get_llm_provider()
    embedder = get_embedder()
    retriever = Retriever(embedder, settings.top_k)
    answer, citations, latency, _tokens = rag_chat(
        session, req.question, top_k=req.top_k, llm=llm, embedder=embedder, retriever=retriever
    )
    return ChatResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency,
        model=f"{settings.llm_provider}:{settings.llm_model}",
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
