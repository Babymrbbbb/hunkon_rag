"""确定性护栏三模式测试：空输入失败 / 同义扩展相关性过滤 / 覆盖路径登记。

背景：
  对应内部「知识库铁律」三条与检索-聚合链路正确性直接相关的通用化护栏
  （不涉及任何真实业务数据，纯方法论）：
    - 模式 B：有效源为空必须显式失败（空输入比报错更危险）
    - 模式 C：相关性过滤先做同义扩展，避免字面匹配误杀
    - 模式 A：口径覆盖闸必须登记在每处实际输出路径
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from app.rules import (  # noqa: E402
    RulingError,
    require_nonempty_sources,
    relevance_filter,
    expand_query,
    verify_ruling_coverage,
)


# ---------------- 模式 B：空输入显式失败 ----------------
def test_require_nonempty_sources_rejects_empty_list():
    with pytest.raises(RulingError):
        require_nonempty_sources([])


def test_require_nonempty_sources_rejects_empty_str():
    with pytest.raises(RulingError):
        require_nonempty_sources("")


def test_require_nonempty_sources_rejects_none():
    with pytest.raises(RulingError):
        require_nonempty_sources(None)


def test_require_nonempty_sources_accepts_nonempty():
    assert require_nonempty_sources(["doc-a"]) is None
    assert require_nonempty_sources("门店资料.md") is None


# ---------------- 模式 C：同义扩展相关性过滤 ----------------
def test_expand_query_includes_synonyms():
    toks = expand_query("复购提升")
    assert "复购" in toks
    assert "回购" in toks  # 同义扩展把「复购提升」映射到「回购」


def test_relevance_filter_keeps_synonym_match():
    """doc 用「回购」写法，但查询是「复购提升」——字面匹配会漏，扩展后应保留。"""
    docs = [
        {"id": "a", "text": "我们通过会员回购提升活动拉动复购。"},
        {"id": "b", "text": "今天天气晴，门店正常营业。"},
    ]
    kept = relevance_filter(docs, "复购提升", required=1)
    ids = [d["id"] for d in kept]
    assert "a" in ids, "同义写法应被召回，不应因字面不匹配被误杀"
    assert "b" not in ids, "无关文档应被过滤掉"


def test_relevance_filter_drops_irrelevant():
    docs = [{"id": "x", "text": "与会员或门店完全无关的内容"}]
    assert relevance_filter(docs, "冷链运输", required=1) == []


# ---------------- 模式 A：覆盖路径登记 ----------------
def test_ruling_coverage_registered():
    """导入 agg_pages / special_pages 后，声明必需覆盖的路径都应已登记。"""
    # 触发两个模块的顶层 register_ruling_path
    import app.agg_pages  # noqa: F401
    import app.special_pages  # noqa: F401

    missing = verify_ruling_coverage()
    assert missing == [], f"以下路径未登记口径覆盖：{missing}"
