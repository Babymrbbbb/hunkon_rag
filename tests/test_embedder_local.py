"""本地 ONNX 向量 provider 的契约测试（provider=local）。

CI 里没有模型文件（613 MB，不进仓库），相关断言自动 skip；
本地有模型时才会真正验证推理与语义区分度。
"""
from pathlib import Path

import pytest

from app.config import settings
from app.embedder import LocalOnnxEmbedder, MockEmbedder, get_embedder
from app.retriever import cosine

MODEL_DIR = Path(settings.embedder_local_dir)
if not MODEL_DIR.is_absolute():
    MODEL_DIR = Path(__file__).resolve().parent.parent / MODEL_DIR
HAS_MODEL = (MODEL_DIR / "tokenizer.json").exists() and any(
    (MODEL_DIR / n).exists() for n in ("model_q4.onnx", "model_quantized.onnx", "model.onnx")
)


def test_mock_fingerprint_includes_dim():
    fp = MockEmbedder().fingerprint
    assert fp == f"mock:{settings.embed_dim}"
    assert fp.startswith("mock:")


def test_local_fingerprint_reflects_actual_implementation():
    """指纹必须反映「实际写库的向量实现」，而不只是模型规格名。

    模型可用时是 `local:<dim>`；模型缺失时编码会降级为 mock，指纹也必须变成
    `mock:<dim>`。否则「降级跑过一次 → 切回真模型时指纹相同 → 不重建索引」，
    库内 256 维旧向量配 1024 维查询，余弦恒 0 且无任何报错（2026-11-03 修）。
    """
    emb = LocalOnnxEmbedder(str(MODEL_DIR))
    if HAS_MODEL:
        assert emb.fingerprint.startswith("local:")
        assert emb.dim > 0
    else:
        assert emb.fingerprint == f"mock:{settings.embed_dim}"
        assert emb.dim == settings.embed_dim


def test_missing_model_fingerprint_falls_back_to_mock():
    """模型路径不存在时，指纹必须如实回落到 mock（不依赖运行环境是否有模型）。"""
    emb = LocalOnnxEmbedder(str(MODEL_DIR / "__not_exist__"))
    assert emb.fingerprint.startswith("mock:")
    assert emb.dim == settings.embed_dim


def test_missing_model_degrades_to_mock():
    """模型缺失时不能抛异常：降级 mock，保证服务可用（与生产 KB 策略一致）。"""
    emb = LocalOnnxEmbedder(str(MODEL_DIR / "__not_exist__"))
    vec = emb.embed("冰面包市场调研")
    assert len(vec) == settings.embed_dim  # 落回 mock 维度


def test_default_provider_from_settings():
    """get_embedder 必须尊重配置（CI 下 conftest 固定为 mock）。"""
    emb = get_embedder()
    provider = settings.embedder_provider
    if provider == "local" and emb.fingerprint.startswith("mock:"):
        pytest.skip("本机缺 local 模型，已降级 mock")
    assert emb.fingerprint.startswith(provider)


@pytest.mark.skipif(not HAS_MODEL, reason="本地未安装 ONNX 模型（CI 环境正常跳过）")
def test_local_embedding_shape():
    emb = LocalOnnxEmbedder(str(MODEL_DIR))
    vec = emb.embed("门店网络覆盖清远、广州、肇庆、佛山")
    assert len(vec) == emb.dim
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-3  # L2 归一化


@pytest.mark.skipif(not HAS_MODEL, reason="本地未安装 ONNX 模型（CI 环境正常跳过）")
def test_local_separates_paraphrase_better_than_mock():
    """真向量应比词袋哈希更懂「换说法」：相关片段与问题的相似度差距要更大。

    这是引入 local 的核心动机——mock 只能匹配字面重叠，「能放几天」与「保质期」
    几乎没有共同字符，语义相近却召回不到。
    """
    question = "冰面包市场调研对象是什么？"
    related = "冰面包非在售，市场通行参考值需 0-4 度冷藏，保质期 3 天。"
    unrelated = "门店网络覆盖清远、广州、肇庆、佛山四地，共 47 家门店。"

    local = LocalOnnxEmbedder(str(MODEL_DIR))
    mock = MockEmbedder()

    def gap(emb):
        qv = emb.embed_query(question)
        return cosine(qv, emb.embed(related)) - cosine(qv, emb.embed(unrelated))

    assert gap(local) > gap(mock), "local 的语义区分度应优于 mock 词袋哈希"
