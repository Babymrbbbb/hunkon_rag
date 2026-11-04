"""零依赖可观测埋点——生产库 langfuse_trace.py 的开源复刻。

能力等价：
  - 把每次 LLM 调用上报到 Langfuse 兼容端点
  - 记录 token 成本、延迟、模型、问题、失败状态
  - 零第三方依赖（urllib + hmac + base64）
  - 采样率控制、失败静默（绝不影响主链路）

与生产库差异：无（纯环境变量驱动，无路径依赖）。

环境变量：
  LANGFUSE_ENABLED      设为 '1' 才启用；默认关
  LANGFUSE_PUBLIC_KEY   公钥（pk-...）
  LANGFUSE_SECRET_KEY   私钥（sk-...）
  LANGFUSE_HOST         Langfuse 地址，默认 https://cloud.langfuse.com
  LANGFUSE_SAMPLE_RATE  采样率 0~1，默认 1.0
  LANGFUSE_DEBUG        设为 '1' 打印上报日志

用法：
  import app.trace as trace
  t = trace.start_trace(name="kb-qa", input_text=..., metadata={...})
  ...
  trace.end_trace(t, output=..., usage=..., latency_ms=..., status="OK")
"""
import base64
import hashlib
import hmac
import json
import os
import random
import sys
import time
import urllib.request


def _enabled() -> bool:
    return os.environ.get("LANGFUSE_ENABLED", "") == "1"


def _sample() -> bool:
    """采样判断：LANGFUSE_SAMPLE_RATE=0.1 时约 10% 请求上报。"""
    try:
        rate = float(os.environ.get("LANGFUSE_SAMPLE_RATE", "1.0"))
    except ValueError:
        rate = 1.0
    if rate >= 1.0:
        return True
    if rate <= 0:
        return False
    return random.random() < rate


def _debug(*args) -> None:
    if os.environ.get("LANGFUSE_DEBUG", "") == "1":
        print("[langfuse]", *args, file=sys.stderr)


def _new_id() -> str:
    return f"{int(time.time() * 1000)}-{random.randrange(16**32):032x}"


def _auth_headers(body_str: str) -> dict:
    """Langfuse Basic 鉴权 + HMAC 签名。"""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    timestamp = str(int(time.time() * 1000))
    basic = base64.b64encode(f"{public_key}:{timestamp}".encode("utf-8")).decode("ascii")
    signature = hmac.new(
        secret_key.encode("utf-8"),
        f"{timestamp}:{body_str}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Basic {basic}",
        "X-Langfuse-Sdk-Signature": signature,
        "X-Langfuse-Sdk-Name": "swan-rag",
        "X-Langfuse-Sdk-Version": "1.0.0",
        "X-Langfuse-Sdk-Variant": "zero-dep-py",
    }


def ingest(events: list) -> None:
    """批量上报事件。任何失败只打日志，绝不影响主链路。"""
    if not _enabled():
        return
    try:
        body = json.dumps({"batch": events}, ensure_ascii=False).encode("utf-8")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        req = urllib.request.Request(
            f"{host}/api/public/ingestion",
            data=body,
            headers=_auth_headers(body.decode("utf-8")),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        _debug("ingest status=200")
    except Exception as e:
        _debug("ingest error=", e)


def start_trace(name: str = "kb-qa", input_text: str = "", metadata: dict = None) -> dict:
    """开始一次 trace：构造 trace-create + generation-create 事件。"""
    if not (_enabled() and _sample()):
        return None
    trace_id = _new_id()
    obs_id = _new_id()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    metadata = metadata or {}
    events = [
        {
            "id": _new_id(),
            "type": "trace-create",
            "timestamp": ts,
            "body": {
                "id": trace_id,
                "name": name,
                "input": input_text or None,
                "metadata": metadata,
                "timestamp": ts,
            },
        },
        {
            "id": _new_id(),
            "type": "generation-create",
            "timestamp": ts,
            "body": {
                "id": obs_id,
                "traceId": trace_id,
                "type": "GENERATION",
                "name": "llm-completion",
                "input": input_text or None,
                "metadata": metadata,
                "startTime": ts,
                "model": metadata.get("model") or None,
            },
        },
    ]
    return {"traceId": trace_id, "observationId": obs_id, "events": events}


def end_trace(
    trace: dict,
    output: str = "",
    usage: dict = None,
    latency_ms: int = 0,
    status: str = "OK",
    error_message: str = "",
) -> None:
    """结束 trace：上报 generation-update + trace 结束。"""
    if not trace:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    obs_id = trace["observationId"]
    trace_id = trace["traceId"]
    gen_end = {
        "id": _new_id(),
        "type": "generation-update",
        "timestamp": ts,
        "body": {
            "id": obs_id,
            "traceId": trace_id,
            "output": output or None,
            "endTime": ts,
            "status": status,
            "level": "ERROR" if status == "ERROR" else "DEFAULT",
            "usage": usage or None,
            "metadata": {"latencyMs": latency_ms},
            "error": {"message": error_message} if error_message else None,
        },
    }
    trace_end = {
        "id": _new_id(),
        "type": "trace-create",
        "timestamp": ts,
        "body": {
            "id": trace_id,
            "timestamp": ts,
            "metadata": {"latencyMs": latency_ms},
        },
    }
    ingest(trace["events"] + [gen_end, trace_end])