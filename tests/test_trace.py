"""可观测埋点测试：事件构造、鉴权头、采样、失败静默。

背景：
  生产知识库 langfuse_trace.py，hunkon_rag 移植为 app/trace.py。
  核心：零依赖上报 LLM 调用到 Langfuse，失败绝不影响主链路。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.trace as trace  # noqa: E402


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    t = trace.start_trace(name="t", input_text="q")
    assert t is None, "默认应禁用"


def test_start_trace_constructs_events(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    t = trace.start_trace(name="kb-qa", input_text="问题", metadata={"model": "glm-5.2"})
    assert t is not None
    assert t["traceId"] and t["observationId"]
    types = [e["type"] for e in t["events"]]
    assert "trace-create" in types
    assert "generation-create" in types
    # generation 带 model
    gen = [e for e in t["events"] if e["type"] == "generation-create"][0]
    assert gen["body"]["model"] == "glm-5.2"
    monkeypatch.delenv("LANGFUSE_ENABLED")


def test_end_trace_ingests(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    sent = []

    def fake_ingest(events):
        sent.append(events)

    monkeypatch.setattr(trace, "ingest", fake_ingest)
    t = trace.start_trace(name="kb-qa", input_text="问题")
    trace.end_trace(t, output="答案", usage={"total": 10}, latency_ms=5, status="OK")
    assert sent, "应触发 ingest"
    all_events = sent[0]
    types = [e["type"] for e in all_events]
    assert "generation-update" in types
    assert "trace-create" in types  # trace 结束
    # generation-update 带 output/status
    gu = [e for e in all_events if e["type"] == "generation-update"][0]
    assert gu["body"]["output"] == "答案"
    assert gu["body"]["status"] == "OK"
    monkeypatch.delenv("LANGFUSE_ENABLED")


def test_auth_headers_contain_required_fields(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    h = trace._auth_headers('{"batch":[]}')
    assert "Authorization" in h and h["Authorization"].startswith("Basic ")
    assert "X-Langfuse-Sdk-Signature" in h
    assert "Content-Type" in h and h["Content-Type"] == "application/json"
    monkeypatch.delenv("LANGFUSE_ENABLED")


def test_sample_rate_zero_disables(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0")
    t = trace.start_trace(name="t", input_text="q")
    assert t is None, "采样率 0 应禁用"
    monkeypatch.delenv("LANGFUSE_ENABLED")


def test_ingest_failure_silent(monkeypatch, capsys):
    """ingest 遇到网络错误应静默，不抛异常。"""
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")  # 不可达

    import app.trace as tr

    tr.ingest([{"id": "x", "type": "trace-create", "timestamp": "t", "body": {}}])
    # 不应抛异常；默认不打印（DEBUG 未开）
    monkeypatch.delenv("LANGFUSE_ENABLED")