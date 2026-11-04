"""特殊页生成测试：dry-run 列出、缺实体页失败、prompt 构造。

背景：
  生产知识库 generate_special_pages.py，hunkon_rag 移植为 app/special_pages.py。
  核心：comparison 对比页 + synthesis 综合页，复用 app.llm 抽象。
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from app import special_pages as sp  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_wiki(monkeypatch):
    """把 wiki 目录指向临时目录，避免测试写入仓库 data/wiki。"""
    tmp = Path(tempfile.mkdtemp()) / "wiki"
    (tmp / "entity").mkdir(parents=True)
    (tmp / "concept").mkdir(parents=True)
    monkeypatch.setattr(sp, "WIKI_DIR", tmp)
    monkeypatch.setattr(sp, "ENTITY_DIR", tmp / "entity")
    monkeypatch.setattr(sp, "CONCEPT_DIR", tmp / "concept")
    monkeypatch.setattr(sp, "COMPARISON_DIR", tmp / "comparison")
    monkeypatch.setattr(sp, "SYNTHESIS_DIR", tmp / "synthesis")
    yield


def test_dry_run_lists_pages(capsys):
    sp.main_entries = None  # 不干扰
    # 直接调用 main 的 dry-run 逻辑
    import io
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["special_pages", "--dry-run"]
    try:
        rc = sp.main()
    finally:
        _sys.argv = old_argv
    assert rc == 0
    out = capsys.readouterr().out
    assert "comparison" in out
    assert "synthesis" in out


def test_gen_comparison_missing_entity():
    ok, res = sp.gen_comparison("A_vs_B", "不存在的实体A", "不存在的实体B")
    assert not ok
    assert "缺少实体页" in res


def test_gen_comparison_mock_success(monkeypatch):
    """mock LLM 回显 prompt 时，应能生成对比页并写入。"""
    # 先构造两个实体页
    (sp.ENTITY_DIR / "Hunkon.md").write_text("# Hunkon\n\n## 摘要\n品牌\n", encoding="utf-8")
    (sp.ENTITY_DIR / "玛芝莲.md").write_text("# 玛芝莲\n\n## 摘要\n竞品\n", encoding="utf-8")

    class FakeLLM:
        def complete(self, system, user):
            return "## 对比维度表\n| 维度 | Hunkon | 玛芝莲 |\n\n## 综合判断\n启示\n\n## 来源列表\n- Hunkon.md\n- 玛芝莲.md\n"

    monkeypatch.setattr("app.special_pages.get_llm_provider", lambda: FakeLLM())
    ok, res = sp.gen_comparison("Hunkon_vs_玛芝莲", "Hunkon", "玛芝莲")
    assert ok, f"mock LLM 应成功: {res}"
    page = sp.COMPARISON_DIR / "Hunkon_vs_玛芝莲.md"
    assert page.exists(), "对比页未写入"
    content = page.read_text(encoding="utf-8")
    assert "Hunkon vs 玛芝莲" in content
    assert "对比维度表" in content


def test_gen_synthesis_mock_success(monkeypatch):
    """mock LLM 回显时，应能生成综合页并写入。"""
    (sp.ENTITY_DIR / "Hunkon.md").write_text("# Hunkon\n\n## 摘要\n品牌\n", encoding="utf-8")

    class FakeLLM:
        def complete(self, system, user):
            return "## 跨源综合\n综合\n\n## 策略建议\n- 建议1\n- 建议2\n\n## 涉及实体/概念\n- Hunkon\n"

    monkeypatch.setattr("app.special_pages.get_llm_provider", lambda: FakeLLM())
    ok, res = sp.gen_synthesis("防御策略", [("Hunkon", "entity")], "Hunkon 防御策略")
    assert ok, f"mock LLM 应成功: {res}"
    page = sp.SYNTHESIS_DIR / "防御策略.md"
    assert page.exists(), "综合页未写入"
    content = page.read_text(encoding="utf-8")
    assert "Hunkon 防御策略" in content
    assert "策略建议" in content


def test_gen_synthesis_no_pages():
    ok, res = sp.gen_synthesis("x", [("不存在", "entity")], "主题")
    assert not ok
    assert "无涉及页面" in res