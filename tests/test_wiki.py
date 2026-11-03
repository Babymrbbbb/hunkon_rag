"""聚合层（wiki）测试：验证 raw→wiki 双轨的聚合页检索。

覆盖：
1. 聚合索引能构建出页面（实体/概念）
2. 检索命中聚合页：问"冰面包"命中 entity/冰面包
3. 聚合页含来源列表与关键事实
4. LLM 编译器：mock LLM 输出可解析为聚合页；解析失败降级规则兜底
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="swan_rag_wiki_test_")
os.environ["RAG_RAG_DB_FILE"] = os.path.join(_TMP, "test.db")
os.environ["RAG_API_KEY"] = "test-secret-key"
os.environ["RAG_LLM_PROVIDER"] = "mock"
os.environ["RAG_EMBEDDER_PROVIDER"] = "mock"
os.environ["RAG_AUTO_SEED"] = "false"
os.environ["RAG_WIKI_COMPILER"] = "rule"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.wiki import LLMWikiIndex, WikiIndex, WikiPage, build_wiki  # noqa: E402


def test_wiki_index_builds_pages():
    wiki = build_wiki()
    assert len(wiki.pages) > 0, "聚合索引为空"
    assert "冰面包" in wiki.pages, f"缺少 entity/冰面包 聚合页，实际: {list(wiki.pages.keys())}"


def test_wiki_search_hit_entity():
    wiki = build_wiki()
    hits = wiki.search("冰面包怎么保存？")
    assert hits, "检索冰面包未命中聚合页"
    assert hits[0].title == "冰面包"
    assert hits[0].kind == "entity"
    assert hits[0].sources, "聚合页缺少来源列表"
    assert hits[0].facts, "聚合页缺少关键事实"


def test_wiki_search_hit_concept():
    wiki = build_wiki()
    hits = wiki.search("加盟政策需要什么条件？")
    assert hits, "检索加盟政策未命中聚合页"
    assert hits[0].title == "加盟政策"
    assert hits[0].kind == "concept"


def test_wiki_save_index():
    """强制构建时聚合索引必须落盘（save_index 默认 force=False 是并发优化：不写盘）。

    注意：data/wiki/ 已被 gitignore（运行时重建），CI 全新检出时文件不存在，
    因此这里必须走 force=True 的真实落盘路径，不能依赖仓库里的残留文件。
    """
    wiki = build_wiki()
    wiki.save_index(force=True)
    index_path = ROOT / "data" / "wiki" / "index.json"
    assert index_path.exists(), "聚合索引文件未生成"


def _mock_llm_page():
    return WikiPage(
        title="测试实体",
        kind="entity",
        summary="这是一个用于测试的聚合页。",
        facts=["关键事实1", "关键事实2"],
        sources=["测试文档"],
    )


def test_llm_compiler_parses_valid_json():
    """mock LLM 返回合法 JSON → LLMWikiIndex 能解析出聚合页。"""
    # 构造 LLMWikiIndex 并注入 mock LLM 输出
    corpus_dir = ROOT / "data" / "corpus"
    llm_index = LLMWikiIndex.__new__(LLMWikiIndex)
    llm_index.corpus_dir = corpus_dir
    llm_index.pages = {}
    llm_index.term_to_page = {}
    llm_index.generated_by = "llm-compiler"
    raw = json.dumps([{
        "title": "冰面包",
        "kind": "entity",
        "summary": "冰面包是夏季爆款。",
        "facts": ["保质期3天"],
        "sources": ["冰面包产品线"],
    }], ensure_ascii=False)
    pages = llm_index._parse_llm_output(raw)
    assert pages, "解析合法 JSON 失败"
    assert pages[0].title == "冰面包"
    assert pages[0].kind == "entity"


def test_llm_compiler_fallback_on_invalid():
    """mock LLM 返回非法输出 → 降级规则兜底，仍产出聚合页。"""
    corpus_dir = ROOT / "data" / "corpus"
    llm_index = LLMWikiIndex.__new__(LLMWikiIndex)
    llm_index.corpus_dir = corpus_dir
    llm_index.pages = {}
    llm_index.term_to_page = {}
    llm_index.generated_by = "llm-compiler"
    pages = llm_index._parse_llm_output("这不是JSON")
    assert pages, "非法输出未降级兜底"
    assert all(isinstance(p, WikiPage) for p in pages)


def test_llm_compiler_survives_llm_exception(monkeypatch):
    """LLM 直接抛异常（429 限流 / 超时 / 断网）时，聚合层必须降级而不是让 /chat 500。

    2026-11-03 修复前：_build_llm 里 llm.complete 没有异常捕获，LLM 一限流就抛到
    /chat 变成 500，整条问答链路不可用。_parse_llm_output 的兜底只覆盖「返回了但
    解析失败」，不覆盖「根本没返回」。本测试锁住后者。
    """

    class BoomLLM:
        def complete(self, *args, **kwargs):
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr("app.wiki.get_llm_provider", lambda: BoomLLM())
    idx = LLMWikiIndex(ROOT / "data" / "corpus")

    assert idx.generated_by == "rule-fallback", "LLM 异常后应标记为规则兜底"
    assert idx.pages, "LLM 不可用时仍必须有规则兜底聚合页，否则聚合层为空"
    assert idx.term_to_page, "兜底后词表映射不能为空，否则聚合检索全失效"