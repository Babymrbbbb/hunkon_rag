"""交叉编码器精排模块——生产库 rag_qa.py 中 Reranker 体系的开源复刻。

能力等价：
  - Reranker 抽象基类：rerank(query, candidates, top_k) -> 重排后列表
  - ScoreReranker：默认实现，基于原分数（实际保持原序，零风险降级）
  - LocalOnnxReranker：本地 ONNX cross-encoder 精排（bge-reranker-base / qwen3-reranker）
    模型缺失/加载失败自动降级 ScoreReranker（不伪造能力）

与生产库差异：
  - 生产库候选是 (doc_id, score)，hunkon_rag 候选是 (score, Chunk)
  - 生产库读文档正文（read_doc_content）；hunkon_rag 直接用 Chunk.text
  - 生产库环境变量 KB_RERANK；hunkon_rag 用 RAG_RERANK

用法：
  from app.reranker import get_reranker
  rr = get_reranker()
  ranked = rr.rerank(question, candidates, top_k)
"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 与生产库 rag_qa.py 保持一致的指令前缀（Qwen3-Reranker 需要 Instruct 格式）
RERANK_INSTRUCT = (
    "Given a question about the Hunkon Noble bakery brand knowledge base, "
    "retrieve relevant passages that answer the question"
)

# 模型目录（相对项目根）：用户通过 scripts/download_reranker.py 下载
RERANKER_MODEL_DIR = _PROJECT_ROOT / "_models" / "bge-reranker-base"
_ONNX_FILENAMES = ("model_q4.onnx", "model_quantized.onnx", "model.onnx")


class Reranker(ABC):
    """精排器抽象基类。"""

    name: str = "base"

    @abstractmethod
    def rerank(self, question: str, candidates: list, top_k: int) -> list:
        """candidates: [(score, Chunk), ...] → 重排后 [(score, Chunk), ...]（取 top_k）。"""


class ScoreReranker(Reranker):
    """默认精排：保持原分数排序（即 RRF 结果原样返回前 top_k）。"""

    name = "score"

    def rerank(self, question: str, candidates: list, top_k: int) -> list:
        # 原样返回前 top_k（candidates 已是 RRF 降序）
        return candidates[:top_k]


class LocalOnnxReranker(Reranker):
    """本地 ONNX cross-encoder 精排。模型缺失/加载失败自动降级 ScoreReranker。"""

    name = "local-cross"

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else RERANKER_MODEL_DIR
        self._np = None
        self._sess = None
        self._tok = None
        self._ok: Optional[bool] = None
        self._fallback = ScoreReranker()

    def _load(self) -> bool:
        if self._ok is not None:
            return self._ok
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer

            onnx_path = None
            for fn in _ONNX_FILENAMES:
                p = self.model_dir / fn
                if p.exists():
                    onnx_path = p
                    break
            tok_path = self.model_dir / "tokenizer.json"
            if not (onnx_path and tok_path.exists()):
                raise FileNotFoundError(f"reranker 模型缺失: {self.model_dir}")

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            outs = {o.name: o.shape for o in sess.get_outputs()}
            # 真正的 cross-encoder 导出：logits 形如 [B, 1] 或 [B, 2]。
            # 若 logits 末维是词表大小（如 151669），说明是按因果 LM 导出的，
            # 分数头不在图里，必须拒收（2026-10-25 生产库实测教训）。
            if "logits" not in outs:
                raise ValueError("该 ONNX 无 logits 输出，不是可用的 cross-encoder 导出")
            last_dim = outs["logits"][-1]
            if not isinstance(last_dim, int) or last_dim > 10:
                raise ValueError(f"logits 末维 {last_dim} 非分数头（词表 LM 导出），拒收")

            self._sess = sess
            self._tok = Tokenizer.from_file(str(tok_path))
            self._tok.enable_truncation(1024)
            self._np = np
            self._ok = True
            print(f"[rerank] 本地 reranker 已加载: {self.model_dir.name}", file=sys.stderr)
        except Exception as e:
            self._ok = False
            print(f"[rerank] 本地 reranker 不可用，降级 ScoreReranker: {e}", file=sys.stderr)
        return self._ok

    def _score_pairs(self, pairs: List[str]) -> list:
        """pairs: [Instruct+Query+Document 文本] → [logit 分数]"""
        np = self._np
        encs = [self._tok.encode(t[:4000]) for t in pairs]
        L = max(len(e.ids) for e in encs)
        B = len(encs)
        ids = np.zeros((B, L), dtype=np.int64)
        mask = np.zeros((B, L), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = e.attention_mask
        feeds = {}
        for inp in self._sess.get_inputs():
            name = inp.name
            if name == "input_ids":
                feeds[name] = ids
            elif name == "attention_mask":
                feeds[name] = mask
            elif name == "token_type_ids":
                feeds[name] = np.zeros((B, L), dtype=np.int64)
            elif name == "position_ids":
                feeds[name] = np.tile(np.arange(L, dtype=np.int64), (B, 1))
            elif name.startswith("past_key_values."):
                shape = [
                    B if d == "batch_size" else (0 if "sequence_length" in str(d) else d)
                    for d in inp.shape
                ]
                feeds[name] = np.zeros(shape, dtype=np.float32)
            else:
                shape = [B if isinstance(d, str) else d for d in inp.shape]
                feeds[name] = np.zeros(shape, dtype=np.int64)
        out = self._sess.run(["logits"], feeds)[0]
        return [float(x) for x in out.reshape(-1)]

    def rerank(self, question: str, candidates: list, top_k: int) -> list:
        if not self._load():
            return self._fallback.rerank(question, candidates, top_k)
        if not candidates:
            return []

        texts = []
        for _score, chunk in candidates:
            t = (chunk.text or "")[:1200].strip()
            texts.append(t if t else f"doc-{chunk.id}")

        pairs = [
            f"Instruct: {RERANK_INSTRUCT}\nQuery: {question}\nDocument: {t}"
            for t in texts
        ]
        try:
            scores = self._score_pairs(pairs)
        except Exception as e:
            print(f"[rerank] 打分失败，降级 ScoreReranker: {e}", file=sys.stderr)
            return self._fallback.rerank(question, candidates, top_k)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [c for c, _s in ranked[:top_k]]


# 进程级单例
_reranker_cache: dict = {}


def get_reranker() -> Reranker:
    """按 settings.reranker 配置返回精排器。

    RAG_RERANK=local 且模型存在 → LocalOnnxReranker（真 cross-encoder）
    其他（默认）→ ScoreReranker（保持 RRF 原序，零风险）
    """
    mode = settings.reranker
    if mode == "local":
        if "local" not in _reranker_cache:
            _reranker_cache["local"] = LocalOnnxReranker()
        return _reranker_cache["local"]
    if "score" not in _reranker_cache:
        _reranker_cache["score"] = ScoreReranker()
    return _reranker_cache["score"]