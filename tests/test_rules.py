"""权威口径覆盖测试：验证 apply_ruling_overrides 的确定性修正。

背景：
  参考实现 kb_rules.py 提供「LLM 生成后确定性覆盖」（裁决写了什么就是什么）。
  hunkon_rag 移植为 app/rules.py，读取 data/store_facts.json + business_facts.json
  作为真相源，对输出做同样修正。

覆盖规则：
  1. 门店总数：40（区域误传）→ 12（权威）
  2. 城市分布：有总数但缺分布时注入
  3. 冰面包：在售/爆款 → 调研对象
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rules import apply_ruling_overrides  # noqa: E402


def test_store_total_corrected_from_region_number():
    """「运营 40 家门店」应被修正为 12 家并加注。"""
    text = "演示品牌 运营 40 家门店，覆盖A 市等地。"
    out = apply_ruling_overrides(text)
    assert "12" in out, f"未修正为权威总数: {out}"
    # 「40」只应出现在澄清说明（外部“40 家”为区域误传），不应作为门店总数
    assert "运营 12 家门店" in out, f"总数未替换: {out}"
    assert "外部“40 家”为区域误传" in out, f"未加口径澄清: {out}"


def test_store_total_kept_when_correct():
    """「全国 12 家门店」应保持不变。"""
    text = "演示品牌 全国 12 家门店。"
    out = apply_ruling_overrides(text)
    assert "12" in out
    assert "区域误传" not in out, f"正确口径被误加注: {out}"


def test_distribution_injected_when_missing():
    """「12 家门店」但缺分布时，注入权威分布。"""
    text = "演示品牌 共有 12 家门店。"
    out = apply_ruling_overrides(text)
    assert "A 市" in out, f"未注入城市分布: {out}"
    assert "B 市" in out and "C 市" in out and "D 市" in out, f"分布不完整: {out}"


def test_distribution_not_duplicated_when_present():
    """正文已含城市分布时，不重复注入。"""
    text = "演示品牌 共有 12 家门店（A 市 6、B 市 3、C 市 2、D 市 1）。"
    out = apply_ruling_overrides(text)
    assert out.count("A 市") == 1, f"分布被重复注入: {out}"


def test_icebread_corrected_from_product():
    """「冰面包是演示品牌 的夏季爆款系列」应被修正为调研对象。"""
    text = "冰面包是演示品牌 的夏季爆款系列，需冷藏保存。"
    out = apply_ruling_overrides(text)
    assert "在售" in out or "调研对象" in out, f"未修正冰面包口径: {out}"
    assert "爆款系列" not in out, f"仍残留爆款表述: {out}"


def test_icebread_kept_when_research():
    """「冰面包是市场调研对象」应保持不变。"""
    text = "冰面包是市场调研对象（非在售）。"
    out = apply_ruling_overrides(text)
    assert "调研对象" in out
    assert "爆款" not in out


def test_empty_text_unchanged():
    assert apply_ruling_overrides("") == ""


def test_unrelated_text_unchanged():
    text = "竞品甲主打干货类产品。"
    assert apply_ruling_overrides(text) == text