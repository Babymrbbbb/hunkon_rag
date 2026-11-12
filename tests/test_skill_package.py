"""rag-ruling 技能包自测：确保三技能注册完整、入口可调用、组合管线端到端正确。"""
from app.rules import (
    RulingError,
    apply_ruling_overrides,
    expand_query,
    relevance_filter,
    require_nonempty_sources,
)
from skills import SKILLS, list_skills, load_skill, run_guard_pipeline


def test_skill_registry_complete():
    names = list_skills()
    assert "rag-ruling/ruling-override" in names
    assert "rag-ruling/query-rewrite" in names
    assert "rag-ruling/source-guard" in names


def test_load_skill_returns_spec():
    spec = load_skill("rag-ruling/source-guard")
    assert spec["mode"] == "B"
    assert "require_nonempty_sources" in spec["entrypoints"]


def test_load_unknown_skill_raises():
    try:
        load_skill("no-such-skill")
        assert False, "应抛出 KeyError"
    except KeyError:
        pass


def test_pipeline_empty_source_fails():
    # 模式 B：空源必须显式失败，而非产出空壳
    try:
        run_guard_pipeline(query="会员复购", docs=[], generated_text="x")
        assert False, "空源应抛 RulingError"
    except RulingError:
        pass


def test_pipeline_synonym_recall():
    # 模式 C：同义扩展召回「回购」写法
    docs = [
        {"text": "会员回购率环比上升"},
        {"text": "门店客流量稳定"},
    ]
    out = run_guard_pipeline(
        query="会员复购提升", docs=docs, generated_text="演示文本"
    )
    kept = out["kept_docs"]
    assert any("回购" in d["text"] for d in kept)
    assert len(kept) == 1  # 客流那条不含任何扩展 token


def test_pipeline_ruling_override_applied():
    # 模式 A：生成文本被确定性覆盖
    text = "演示品牌共 40 家门店"
    out = run_guard_pipeline(
        query="门店分布",
        docs=[{"text": "门店总数"}],
        generated_text=text,
        page_name="演示品牌",
    )
    assert "40 家" not in out["guarded_text"] or "权威口径" in out["guarded_text"]


def test_entrypoints_are_callable():
    for spec in SKILLS.values():
        for ep in spec["entrypoints"]:
            assert callable(globals().get(ep) or getattr(__import__("app.rules", fromlist=[ep]), ep))
