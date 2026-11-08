"""权威口径确定性覆盖（共享闸模块）——参考实现 kb_rules.py 的开源复刻。

背景：
  参考实现用 kb_rules.apply_ruling_overrides() 在「LLM 生成之后」做确定性覆盖：
  裁决文档写了什么就是什么，不靠 LLM 自觉。hunkon_rag 的真相源是
  data/business_facts.json + data/store_facts.json（JSON 版裁决文档），
  本模块读取它们，对聚合页/LLM 输出做同样的确定性后处理。

与参考实现的差异：
  - 参考实现读「裁决 .md 文档」（路径硬编码 KB 根目录）
  - hunkon_rag 读「真相源 JSON」（相对路径，随仓库分发，clone 后替换 JSON 即可）
  - 覆盖规则由 JSON 内容驱动，不写死业务数字（除「冰面包非在售」这类权威口径外）

用法：
  from app.rules import apply_ruling_overrides
  text = apply_ruling_overrides(text, page_name="演示品牌")
"""
import json
import re
from pathlib import Path

# 真相源路径（相对项目根，随仓库分发）
ROOT = Path(__file__).resolve().parent.parent
STORE_FACTS = ROOT / "data" / "store_facts.json"
BUSINESS_FACTS = ROOT / "data" / "business_facts.json"

# 权威口径：冰面包并非演示品牌的在售产品，仅是市场调研对象（合成演示设定）。
# 这是 hunkon_rag 特有的演示裁决，无法从 JSON 推导，因此作为常量规则。
ICEBREAD_RULING = (
    "冰面包并非演示品牌的在售产品，仅是市场调研对象（合成演示设定，未上线）"
)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _store_facts() -> dict:
    return _load_json(STORE_FACTS)


def _business_facts() -> dict:
    return _load_json(BUSINESS_FACTS)


def apply_ruling_overrides(text: str, page_name: str = "") -> str:
    """确定性后处理：用真相源覆盖 LLM/规则输出中的错误口径。

    规则（全部为确定性替换，不做任何推理）：
    1. 门店总数：把「运营/现有/共/拥有/全国 N 家门店」中的 N 修正为真相源总数，
       且当 N 与真相源不符（被裁决否定的旧口径）时，加注澄清。
    2. 城市分布：若正文出现门店总数但缺少权威分布，注入分布。
    3. 冰面包：若正文把冰面包写成在售产品/爆款系列，强制修正为调研对象。
    """
    if not text:
        return text

    out = text
    facts = _store_facts()
    total = facts.get("total_stores")
    distribution = facts.get("distribution", {})

    # 规则 1：门店总数修正
    if total:
        for pat in (
            r"运营\s*(\d+)\s*家门店",
            r"现有\s*(\d+)\s*家门店",
            r"共\s*(\d+)\s*家门店",
            r"拥有\s*(\d+)\s*家门店",
            r"全国\s*(\d+)\s*家门店",
        ):
            def _fix(m: re.Match) -> str:
                n = int(m.group(1))
                if n == total:
                    return m.group(0)
                return m.group(0).replace(str(n), str(total)) + (
                    f"（权威口径：{total} 家，外部“{n} 家”为区域误传）"
                )

            out = re.sub(pat, _fix, out)

    # 规则 2：若出现权威总数但缺少城市分布，注入权威分布
    if total and distribution and f"{total} 家" in out:
        if not any(city.split("（")[0] in out for city in distribution):
            dist_text = "，".join(
                f"{city.split('（')[0]} {n}" for city, n in distribution.items()
            )
            out = out.replace(
                f"{total} 家",
                f"{total} 家（{dist_text}）",
                1,
            )

    # 规则 3：冰面包权威口径
    if "冰面包" in out:
        if re.search(r"冰面包.*?(在售|爆款|产品线|主打口味)", out) or re.search(
            r"(在售|爆款|产品线).*?冰面包", out
        ):
            out = re.sub(
                r"冰面包(是|为)(演示品牌\s*的)?(夏季爆款系列|在售产品|产品线)",
                ICEBREAD_RULING,
                out,
            )
            if "调研对象" not in out:
                out = out + f"\n\n> ⚠️ {ICEBREAD_RULING}。"

    return out
