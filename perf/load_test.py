"""性能压测 · /chat 接口并发压测脚本

用 Python 标准库并发请求 hunkon_rag 的 /chat 接口，统计：
- P50 / P95 / P99 延迟
- QPS（每秒请求数）
- 错误率

用法：
    # 先启动服务（mock 模式，避免上游限流影响压测）：
    #   RAG_LLM_PROVIDER=mock RAG_WIKI_COMPILER=rule python -m uvicorn app.main:app --port 8000
    #
    # 压测：
    python perf/load_test.py --url http://127.0.0.1:8000/chat --concurrency 100 --requests 300

产物：
    perf/results/load_test_<时间戳>.json  —— 原始结果
    perf/results/load_test_<时间戳>.md    —— Markdown 报告
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_QUESTION = "演示品牌 目前一共有多少家门店，主要在哪些城市？"


def send_one(url: str, api_key: str, question: str, timeout: int) -> tuple[int, float, int]:
    """发送一次 /chat 请求。返回 (HTTP状态码, 延迟ms, 是否错误)。"""
    body = json.dumps({"question": question, "top_k": 3}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = 0  # 连接失败/超时
    latency_ms = (time.perf_counter() - t0) * 1000
    return status, latency_ms, status not in (200,)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * p)))
    return sorted_data[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/chat")
    ap.add_argument("--api-key", default="dev-secret-key")
    ap.add_argument("--concurrency", type=int, default=100, help="并发数")
    ap.add_argument("--requests", type=int, default=300, help="总请求数")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    args = ap.parse_args()

    latencies: list[float] = []
    errors = 0
    status_counts: dict[int, int] = {}
    lock = threading.Lock()

    def worker(_: int) -> tuple[int, float, int]:
        status, lat, is_err = send_one(args.url, args.api_key, args.question, args.timeout)
        with lock:
            latencies.append(lat)
            status_counts[status] = status_counts.get(status, 0) + 1
        return status, lat, is_err

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, i) for i in range(args.requests)]
        for fut in as_completed(futures):
            _, _, is_err = fut.result()
            if is_err:
                errors += 1
    total_time = time.time() - t0

    qps = args.requests / total_time if total_time > 0 else 0
    ok_count = args.requests - errors
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "url": args.url,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "total_time_s": round(total_time, 2),
        "qps": round(qps, 2),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "error_rate": round(errors / args.requests, 4),
        "ok_count": ok_count,
        "status_counts": status_counts,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"load_test_{ts}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / f"load_test_{ts}.md").write_text(
        _to_markdown(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _to_markdown(r: dict) -> str:
    return f"""# 性能压测报告 · {r['timestamp']}

- 目标: {r['url']}
- 并发: {r['concurrency']}
- 请求数: {r['requests']}
- 总耗时: {r['total_time_s']}s

## 指标

| 指标 | 值 |
|---|---|
| QPS | {r['qps']} |
| P50 延迟 | {r['p50_ms']} ms |
| P95 延迟 | {r['p95_ms']} ms |
| P99 延迟 | {r['p99_ms']} ms |
| 平均延迟 | {r['avg_ms']} ms |
| 错误率 | {r['error_rate']:.2%} |
| 成功请求 | {r['ok_count']} / {r['requests']} |

## 状态码分布

```json
{json.dumps(r['status_counts'], ensure_ascii=False, indent=2)}
```
"""


if __name__ == "__main__":
    raise SystemExit(main())