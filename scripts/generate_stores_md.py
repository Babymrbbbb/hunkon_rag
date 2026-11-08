"""从 data/store_facts.json（唯一真相源）重新生成 data/corpus/01_stores.md。

用途：门店数据更新时，只需改 store_facts.json，再跑本脚本即可让门店网络文件与真相源
严格一致，避免手写漂移。这是「单一真相源 → 生成视图」的关键环节。
用法：
    python scripts/generate_stores_md.py
"""
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = ROOT / "data" / "store_facts.json"
OUT_PATH = ROOT / "data" / "corpus" / "01_stores.md"

facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
total = facts["total_stores"]
dist = facts["distribution"]

# 核心市场 = 门店数最多者（动态标注，不写死）
core = max(dist, key=lambda k: dist[k])

parts = []
for city, n in dist.items():
    if city == core:
        parts.append(f"{city} {n} 家为密度最高的核心市场")
    else:
        parts.append(f"{city} {n} 家")
body = "，".join(parts)

text = (
    f"# 门店网络\n"
    f"演示品牌 目前在全国拥有 {total} 家门店，其中 {body}。"
    f"所有门店采用一致的视觉与陈列规范，支持小程序下单与到店自提。"
    f"拓店计划属于合成演示设定，具体节奏与区域不在语料中给出固定数字。"
    f"城市分布与会员规模以唯一真相源 data/store_facts.json 为准。\n"
    f"\n> 合成演示数据说明：本仓库数据为合成演示数据，与任何真实企业经营数据无关；文档只描述方法论与工程实现。\n"
)

OUT_PATH.write_text(text, encoding="utf-8")
print(f"已根据 store_facts.json 重新生成 {OUT_PATH}")
print(f"  门店总数={total}，核心市场={core}（{dist[core]} 家）")
