"""聚合页 LLM 生成测试：扫描、类型判断、截断检测、prompt 构建、生成链路。

背景：
  生产知识库 generate_agg_pages.py，hunkon_rag 移植为 app/agg_pages.py。
  核心：用 LLM 批量生成聚合页，并带权威口径覆盖 + 截断检测。

注意：scan_sources 的聚合页键名是「文档完整标题」（如「品牌介绍：演示品牌」），
不是裸实体名（如「演示品牌」）。这与参考实现 aggregation_index 的 page 名一致。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agg_pages import (  # noqa: E402
    REQUIRED_FIELDS,
    build_prompt,
    detect_truncated,
    fix_type,
    scan_sources,
)

CORPUS = ROOT / "data" / "corpus"

BRAND_PAGE = "品牌介绍：演示品牌"
STORE_PAGE = "门店网络"


def test_scan_sources_finds_pages():
    agg = scan_sources(CORPUS)
    assert BRAND_PAGE in agg, f"缺少品牌页: {list(agg.keys())[:10]}"
    assert STORE_PAGE in agg, "缺少门店网络页"
    assert "冰面包市场调研" in agg, "缺少冰面包调研页"
    # 每页至少一个源
    for page, info in agg.items():
        assert info["sources"], f"{page} 无源"


def test_scan_sources_type_entity_for_brand():
    agg = scan_sources(CORPUS)
    assert agg[BRAND_PAGE]["type"] == "concept"  # 品牌页标题含「品牌介绍」，非裸品牌名
    # 通过 fix_type 可将品牌页修正为 entity
    assert fix_type("演示品牌", agg[BRAND_PAGE]["type"]) == "entity"


def test_scan_sources_cross_references():
    """正文包含其它标题词的文档应挂到对应页。"""
    agg = scan_sources(CORPUS)
    # 01_stores.md 标题是「门店网络」，正文含「演示品牌」→ 应挂到品牌页
    brand_sources = agg[BRAND_PAGE]["sources"]
    assert len(brand_sources) >= 2, f"品牌页应含跨引用源: {brand_sources}"


def test_fix_type():
    assert fix_type("演示品牌", "concept") == "entity"
    assert fix_type("竞品乙", "concept") == "entity"
    assert fix_type("门店网络", "entity") == "entity"  # 非品牌保持原类型
    assert fix_type("供应链", "concept") == "concept"


def test_detect_truncated():
    ok_text = "# 测试\n\n## 摘要\n内容\n\n## 关键事实\n事实\n\n## 来源\n源1\n"
    truncated, missing = detect_truncated(ok_text, "entity")
    assert not truncated, f"完整文本被误判截断: {missing}"

    # 「没有关键事实」是句子而非小节标题，不应算已含字段
    bad_text = "# 测试\n\n## 摘要\n没有关键事实和来源\n"
    truncated, missing = detect_truncated(bad_text, "entity")
    assert truncated, "缺字段未被判定截断"
    assert "关键事实" in missing, f"缺失字段识别错误: {missing}"
    assert "来源" in missing, f"缺失字段识别错误: {missing}"


def test_required_fields_cover_both_types():
    assert "摘要" in REQUIRED_FIELDS["entity"]
    assert "摘要" in REQUIRED_FIELDS["concept"]
    assert "概念描述" in REQUIRED_FIELDS["concept"]


def test_build_prompt_includes_essential():
    prompt = build_prompt("门店网络", "concept", "源内容", ["01_stores.md"])
    assert "门店网络" in prompt
    assert "concept" in prompt
    assert "01_stores.md" in prompt, f"源路径未列出: {prompt}"
    assert "摘要" in prompt and "来源" in prompt
    assert "严禁编造" in prompt, "缺少防编造指令"