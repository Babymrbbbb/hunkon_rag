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


# ===========================================================================
# 确定性护栏三模式（方法论开源复刻，对应内部「知识库铁律」的通用化）
# ---------------------------------------------------------------------------
# 内部工程总结的 11 条铁律里，有三条与「检索 / 聚合链路的正确性」直接相关，
# 且不依赖任何真实业务数据，可抽象成通用护栏复刻到本仓库（纯方法论，合成演示）：
#   模式 A（铁律·口径覆盖必须落在每处执行路径）：凡「LLM 生成后确定性覆盖」的闸，
#           必须在所有实际输出路径上 import 复用同一份函数——只改一处 = 形同虚设。
#   模式 B（铁律·空输入比报错更危险）：聚合 / 生成入口若收到的有效源为空，
#           必须显式失败，绝不允许拿 0 个源去问 LLM 产出「空壳页」过校验。
#   模式 C（铁律·相关性过滤须做同义扩展）：字面匹配会误杀相关文档
#           （如「复购提升」字面搜不到「回购」），过滤前必须先做同义扩展。
# ===========================================================================


class RulingError(ValueError):
    """护栏违反：空输入、覆盖缺失、相关性过滤异常等。"""


# 演示同义词表（合成，与任何真实业务无关）：用于模式 C 的同义扩展。
# 内部实战里「复购提升」因字面匹配被误杀、召回从 25 掉到 2；
# 这里用通用零售术语演示同一机制，不涉及真实品牌 / 门店。
SYNONYMS = {
    "复购": ["复购", "回购", "复购率", "复购提升", "复购行为"],
    "会员": ["会员", "会员数", "会员规模", "会员量", "会员体系"],
    "新品": ["新品", "限定新品", "夏季新品", "新品上市"],
    "冷链": ["冷链", "冷链温控", "全程冷链", "冷链运输"],
    "门店": ["门店", "店面", "店铺", "分店"],
}


def expand_query(query: str) -> set:
    """把查询里的每个词做同义扩展，返回扩展后的 token 集合（模式 C 前置步骤）。"""
    if not query:
        return set()
    tokens: set = set()
    for canon, syns in SYNONYMS.items():
        if canon in query:
            tokens.update(syns)
    # 也把查询原词拆进集合（按空白 / 常见标点切分）
    for piece in re.split(r"[\s，。、；：,.;:]+", query):
        if piece:
            tokens.add(piece)
    return tokens


def relevance_filter(docs: list, query: str, required: int = 1) -> list:
    """按「同义扩展后的 token」过滤相关文档（模式 C）。

    docs: 元素为 dict（含 text 字段，可选 id）；或直接为 list[str]。
    返回：命中度 >= required 的文档（保持原顺序）。
    说明：纯字面匹配会漏掉同义表述；先 expand_query 再按 token 命中，
    避免「复购提升」类同义写法被误杀（内部实战同义召回 25 → 2 的教训）。
    """
    tokens = expand_query(query)
    if not tokens:
        return list(docs)
    kept = []
    for d in docs:
        text = d.get("text", d) if isinstance(d, dict) else d
        text = text or ""
        hits = sum(1 for t in tokens if t and t in text)
        if hits >= required:
            kept.append(d)
    return kept


def require_nonempty_sources(sources) -> None:
    """模式 B：聚合 / 生成入口的「有效源数 ≥ 1」前置校验。

    内部实战教训：拿 0 个源去问 LLM 会产出空壳页且能过字段校验，
    覆盖掉完整旧版。故必须显式失败，而不是静默继续（空输入比报错更危险）。
    """
    try:
        empty = len(sources) == 0
    except TypeError:
        empty = not sources
    if empty:
        raise RulingError("有效源为空：不允许以 0 个源生成页面（空壳页风险）")
    return None


# ---- 模式 A：口径覆盖路径登记（确保闸落在每处执行路径）----
RULING_ENFORCED_PATHS: set = set()
REQUIRED_RULING_PATHS = (
    "agg_pages.llm_agg",
    "special_pages.gen_comparison",
    "special_pages.gen_synthesis",
)


def register_ruling_path(path: str) -> None:
    """登记一条「已对输出应用口径覆盖」的执行路径。

    凡实际产出对外文本的函数，都应在 apply_ruling_overrides 调用处登记自身，
    使 verify_ruling_coverage() 能在 CI 中证明「闸落在了每处路径」，
    而不是只改了一处（形同虚设）。
    """
    RULING_ENFORCED_PATHS.add(path)


def verify_ruling_coverage() -> list:
    """返回「声明为必需覆盖、却尚未登记」的执行路径（空列表 = 通过）。"""
    return [p for p in REQUIRED_RULING_PATHS if p not in RULING_ENFORCED_PATHS]
