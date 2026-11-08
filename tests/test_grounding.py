"""数据接地护栏：确保知识库 corpus 与权威真相源一致，阻断错误资料回流。

设计原则（数据无关，可随业务更新而长期有效）：
- store_facts.json 是门店数据的唯一真相源（Single Source of Truth），也是**唯一更新入口**。
- 本测试不写死任何具体数字/城市，只校验「corpus ↔ 真相源」的一致性：
  1) 真相源自洽（各城市之和 = 总数）；
  2) corpus 中所有「X 家门店」表述必须等于真相源总数（防 stale / 矛盾数字）；
  3) corpus 必须覆盖真相源每个城市及其数量；
  4) corpus 不得出现真相源列为不存在的城市（防虚构，如曾凭空写的『E 市』）。
- 更新数据流程：改 store_facts.json → 跑 scripts/generate_stores_md.py（重建 01_stores.md）
  → 改其余 corpus 叙述使之吻合 → pytest 通过即说明接地一致 → 启动自愈合重建索引。
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
FACTS_PATH = ROOT / "data" / "store_facts.json"

facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def _corpus_text() -> str:
    return "\n".join(
        f.read_text(encoding="utf-8") for f in sorted(CORPUS_DIR.glob("*.md"))
    )


def test_truth_source_self_consistent():
    """真相源自身须自洽：各城市之和 = 总数。"""
    assert sum(facts["distribution"].values()) == facts["total_stores"]


def test_no_fabricated_city():
    """corpus 不得出现真相源标记为不存在的城市（如曾凭空写入的『E 市』）。"""
    text = _corpus_text()
    for city in facts["nonexistent_cities"]:
        assert city not in text, f"corpus 仍含虚构城市『{city}』，须从真相源剔除"


def test_store_total_consistent():
    """corpus 中所有『X 家门店』表述（拥有/全国）必须与真相源总数一致。

    用『拥有/全国 X 家门店』两种前缀定位「门店总数」表述，排除拓店计划等干扰描述，
    从而能抓出任一文件里的 stale / 矛盾门店数（如某文件忘了随真相源更新）。
    """
    text = _corpus_text()
    totals: list[int] = []
    for pat in (r"拥有\s*(\d+)\s*家门店", r"全国\s*(\d+)\s*家门店"):
        totals += [int(m) for m in re.findall(pat, text)]
    assert totals, "corpus 未出现任何『X 家门店』表述"
    assert all(t == facts["total_stores"] for t in totals), (
        f"corpus 出现与真相源不符的门店总数: {sorted(set(totals))} "
        f"（真相源={facts['total_stores']}）"
    )


def test_distribution_matches_truth():
    """corpus 必须覆盖真相源中每个城市及其数量。"""
    text = _corpus_text()
    for city, n in facts["distribution"].items():
        base = city.split("（")[0]  # 容忍「A 市」等带括号写法
        assert base in text, f"corpus 缺失城市『{city}』"
        assert str(n) in text, f"corpus 缺失城市『{city}』的数量 {n}"
