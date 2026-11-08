"""轻量可观测：进程内指标收集（架构师视角的可观测性最小实现）。"""
import threading


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.chat_requests = 0
        self.latencies: list = []
        self.total_tokens = 0

    def record_request(self) -> None:
        with self._lock:
            self.requests += 1

    def record_chat(self, latency_ms: int, tokens: int = 0) -> None:
        with self._lock:
            self.chat_requests += 1
            self.latencies.append(latency_ms)
            self.total_tokens += tokens

    @property
    def avg_latency(self) -> float:
        with self._lock:
            return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


metrics = Metrics()
