"""向量化（Embedder）抽象层。

三档 provider，同一接口（ADR-001 续）：
- mock  ：字符/二元组哈希到定长向量，纯 Python、离线、确定性，用于 CI 与单元测试。
- local ：本地 ONNX 量化模型（qwen3-embedding-0.6b 1024 维 / bge-small-zh 512 维），
          CPU 推理、数据不出机；模型或 onnxruntime/tokenizers 缺失时自动降级 mock。
- real  ：OpenAI 兼容 /embeddings 接口。

为什么要有 local：mock 是词袋哈希，只有「字面重叠」近似语义，同义改写（如
「保质期」↔「能放几天」）召回不到。local 走真语义向量，与生产知识库
（D:/WorkBuddy知识库/build_vector_index.py）同一套模型与池化参数。

维度变更必须重建索引：seed.py 会把 fingerprint（模型名:维度）写进 Meta 表，
不一致时自动重建，避免「256 维旧库 + 1024 维查询」导致余弦恒为 0 的静默失效。
"""
import hashlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 高频中文虚词/标点：mock 向量过滤它们，让内容词主导相似度（贴近真实嵌入语义）。
_STOP = set(
    "的是在有着与和以为其也将并而之及等个们都可所被把要对中内上下面后前这那"
    "或就也很更最再没无此从向当该各它她你我又但因若由、，。：；！？（）《》"
    "「」*_~#@%&$=+-/\\| "
)


def _tokens(text: str) -> list:
    """中英文通用的确定性分词：单字 + 二元组，过滤高频虚词与标点。"""
    text = text.lower()
    chars = [c for c in text if c not in _STOP]
    toks = chars[:]
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


class Embedder(ABC):
    """向量化接口。dim 与 fingerprint 用于「换模型必须重建索引」的一致性校验。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> list:
        """把文档片段编码为定长向量（写库时用）。"""

    def embed_query(self, text: str) -> list:
        """把用户问题编码为向量。默认与文档同编码；非对称模型（如 qwen3）需覆写加查询前缀。"""
        return self.embed(text)

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.dim}"


class MockEmbedder(Embedder):
    name = "mock"

    def __init__(self) -> None:
        self.dim = settings.embed_dim

    def embed(self, text: str) -> list:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class RealEmbedder(Embedder):
    """OpenAI 兼容 embeddings 接口（provider=real）。"""

    name = "real"

    def __init__(self) -> None:
        self.base_url = settings.embedder_base_url.rstrip("/")
        self.api_key = settings.embedder_api_key or settings.llm_api_key
        self.model = settings.embedder_model
        self.dim = settings.embed_dim

    def embed(self, text: str) -> list:
        if not self.api_key:
            raise RuntimeError("embedder_api_key 未配置，无法调用真实向量化；请设置 RAG_EMBEDDER_API_KEY")
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = resp.json()["data"][0]["embedding"]
        self.dim = len(vec)
        return vec

    @property
    def fingerprint(self) -> str:
        return f"real:{self.model}:{self.dim}"


# 本地 ONNX 模型规格：与生产知识库 build_vector_index.py 的 LOCAL_MODEL_CANDIDATES 保持一致。
# 两处参数必须同步修改，否则同一段文本在两套系统里会得到不同向量。
_LOCAL_SPECS = {
    "qwen3-embedding-0.6b": {
        "dim": 1024,
        "pooling": "last",  # Qwen3 官方用法：取最后一个有效 token
        "query_prompt": (
            "Instruct: Given a question about the Hunkon Noble bakery brand "
            "knowledge base, retrieve relevant passages that answer the question\nQuery: "
        ),
        "doc_prompt": "Document: ",
        "max_len": 256,
    },
    "bge-small-zh": {
        "dim": 512,
        "pooling": "cls",  # BGE 官方用法：CLS pooling，无前缀
        "query_prompt": "",
        "doc_prompt": "",
        "max_len": 512,
    },
}
_DEFAULT_SPEC = _LOCAL_SPECS["qwen3-embedding-0.6b"]
_ONNX_FILENAMES = ("model_q4.onnx", "model_quantized.onnx", "model.onnx")


class LocalOnnxEmbedder(Embedder):
    """本地量化 ONNX 模型推理（provider=local）。

    惰性加载：首次 embed 才载入模型（613 MB，冷启动数秒），避免拖慢进程启动。
    加载失败（缺模型/缺依赖）自动降级 MockEmbedder，服务不中断——与生产知识库
    build_vector_index.py 的降级策略一致。
    """

    name = "local"

    def __init__(self, model_dir: Optional[str] = None) -> None:
        raw = Path(model_dir or settings.embedder_local_dir)
        self.model_dir = raw if raw.is_absolute() else (_PROJECT_ROOT / raw)
        self.spec = _LOCAL_SPECS.get(self.model_dir.name, _DEFAULT_SPEC)
        self._spec_dim = int(self.spec["dim"])
        self._np = None
        self._sess = None
        self._tok = None
        self._ok: Optional[bool] = None
        self._fallback = MockEmbedder()

    # ---------- 一致性元信息 ----------
    # 必须反映「实际写库的向量」，不能只报模型规格：模型缺失时编码走 mock，
    # 落的库是 256 维；若指纹仍写 local:1024，下次切回真模型指纹相同便不会重建，
    # 于是 256 维旧库 + 1024 维查询，余弦恒为 0 且无任何报错。

    @property
    def dim(self) -> int:  # type: ignore[override]
        if self._ok is None:
            self._load()
        return self._spec_dim if self._ok else self._fallback.dim

    @property
    def fingerprint(self) -> str:  # type: ignore[override]
        if self._ok is None:
            self._load()
        return f"local:{self._spec_dim}" if self._ok else self._fallback.fingerprint

    # ---------- 内部 ----------

    def _onnx_path(self) -> Optional[Path]:
        for fn in _ONNX_FILENAMES:
            p = self.model_dir / fn
            if p.exists():
                return p
        return None

    def _load(self) -> bool:
        if self._ok is not None:
            return self._ok
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer

            onnx_path = self._onnx_path()
            tok_path = self.model_dir / "tokenizer.json"
            if not onnx_path or not tok_path.exists():
                raise FileNotFoundError(f"本地模型文件缺失: {self.model_dir}")
            self._sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._tok = Tokenizer.from_file(str(tok_path))
            self._tok.enable_truncation(self.spec.get("max_len", 512))
            self._np = np
            self._ok = True
            print(f"[embed] local 模型已加载: {self.model_dir.name} ({self._spec_dim} 维)", file=sys.stderr)
        except Exception as e:  # 缺依赖、模型损坏、路径错误 —— 一律降级，不阻断服务
            self._ok = False
            print(f"[embed] 本地模型不可用，降级 mock: {e}", file=sys.stderr)
        return self._ok

    def _encode(self, text: str, is_query: bool) -> list:
        if not self._load():
            return self._fallback.embed(text)
        np = self._np
        prompt = self.spec.get("query_prompt" if is_query else "doc_prompt") or ""
        enc = self._tok.encode((prompt + text)[:8000])
        length = len(enc.ids)

        # 按会话输入清单动态构造 feeds：兼容编码器式（BGE）与解码器式（Qwen3 KV-cache 导出）
        feeds = {}
        for inp in self._sess.get_inputs():
            name = inp.name
            if name == "input_ids":
                feeds[name] = np.array([enc.ids], dtype=np.int64)
            elif name == "attention_mask":
                feeds[name] = np.array([enc.attention_mask], dtype=np.int64)
            elif name == "token_type_ids":
                feeds[name] = np.array([enc.type_ids], dtype=np.int64)
            elif name == "position_ids":
                feeds[name] = np.arange(length, dtype=np.int64)[None, :]
            elif name.startswith("past_key_values."):
                shape = [
                    1 if d == "batch_size" else (0 if "sequence_length" in str(d) else d)
                    for d in inp.shape
                ]
                feeds[name] = np.zeros(shape, dtype=np.float32)  # past=0 → 全序列一次前向
            else:
                shape = [1 if isinstance(d, str) else d for d in inp.shape]
                feeds[name] = np.zeros(shape, dtype=np.int64)

        out_names = [o.name for o in self._sess.get_outputs()]
        key = ["last_hidden_state"] if "last_hidden_state" in out_names else None
        out = self._sess.run(key, feeds)[0]

        if self.spec.get("pooling") == "last":
            last = int(sum(enc.attention_mask)) - 1
            vec = out[0, last]
        else:
            vec = out[0, 0]  # CLS pooling

        norm = float(np.linalg.norm(vec))
        vec = vec / norm if norm else vec
        return [float(x) for x in vec]

    # ---------- 接口 ----------

    def embed(self, text: str) -> list:
        return self._encode(text, is_query=False)

    def embed_query(self, text: str) -> list:
        return self._encode(text, is_query=True)


# 进程级单例缓存：避免每次 /chat 都新建 embedder（2026-10-30 压测优化）。
_embedder_cache: dict = {}


def get_embedder() -> Embedder:
    provider = settings.embedder_provider
    # 缓存键对 mock 带上维度：MockEmbedder 的 dim 在构造时取自 settings.embed_dim，
    # 只按 provider 缓存会导致「改了维度仍拿回旧实例」，指纹谎报旧维度。
    cache_key = f"{provider}:{settings.embed_dim}" if provider == "mock" else provider
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]
    if provider == "real":
        embedder: Embedder = RealEmbedder()
    elif provider == "local":
        embedder = LocalOnnxEmbedder()
    else:
        embedder = MockEmbedder()
    _embedder_cache[cache_key] = embedder
    return embedder
