"""特殊页生成（comparison 对比页 / synthesis 综合页）——生产库 generate_special_pages.py 的开源复刻。

能力等价：
  - comparison：基于两个实体页，生成对比维度表 + 综合判断
  - synthesis：基于多个相关实体/概念页，生成跨源综合 + 策略建议

与生产库差异：
  - 生产库用 agnes-2.5-flash 直连 + 硬编码绝对路径
  - swan_rag 复用 app.llm.get_llm_provider()（LLMProvider 抽象）
  - 生产库读 _meta/wiki_schema.md；swan_rag 用内置 prompt 模板

用法：
  python -m app.special_pages [--dry-run]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import get_llm_provider  # noqa: E402

WIKI_DIR = ROOT / "data" / "wiki"
ENTITY_DIR = WIKI_DIR / "entity"
CONCEPT_DIR = WIKI_DIR / "concept"
COMPARISON_DIR = WIKI_DIR / "comparison"
SYNTHESIS_DIR = WIKI_DIR / "synthesis"

_SYSTEM_PROMPT = (
    "你是企业知识库的 LLM Wiki 编译引擎。给定品牌实体页或多个相关页面，"
    "生成对比页（comparison）或综合页（synthesis）。输出纯 Markdown，不要 JSON。"
)


def read_page(page_path: Path) -> str:
    try:
        return page_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def gen_comparison(page_name: str, entity_a: str, entity_b: str) -> tuple:
    """生成对比页：A vs B。返回 (ok, result)。"""
    pa = ENTITY_DIR / f"{entity_a}.md"
    pb = ENTITY_DIR / f"{entity_b}.md"
    if not pa.exists() or not pb.exists():
        return False, f"缺少实体页: {entity_a} 或 {entity_b}"
    content_a = read_page(pa)[:5000]
    content_b = read_page(pb)[:5000]

    prompt = f"""【本次任务】基于以下两个品牌实体页，生成一张 comparison 对比页：{entity_a} vs {entity_b}。

要求：
1. 「对比维度表」至少 6 个维度（定位/价格带/门店规模/核心品类/营销打法/威胁等级等）。
2. 「综合判断」给出对天鹅公主的竞争启示。
3. 「来源列表」引用两个实体页各自的主要来源。
4. 「原文引用」从两个实体页中各取关键句。
5. 输出纯 Markdown。

【实体A：{entity_a}】
{content_a}

【实体B：{entity_b}】
{content_b}
"""
    try:
        llm = get_llm_provider()
        compiled = llm.complete(_SYSTEM_PROMPT, prompt)
    except Exception as e:
        return False, f"LLM 调用失败: {e}"
    if not compiled or not compiled.strip():
        return False, "LLM 返回空输出"

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    page_path = COMPARISON_DIR / f"{page_name}.md"
    content = (
        f"# {entity_a} vs {entity_b}\n\n"
        f"- 类型：comparison\n"
        f"- 编译时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{compiled.strip()}\n"
    )
    page_path.write_text(content, encoding="utf-8")
    return True, str(page_path)


def gen_synthesis(page_name: str, involved: list, strategy_topic: str) -> tuple:
    """生成综合页：跨源综合 + 策略建议。返回 (ok, result)。"""
    parts = []
    for name, ptype in involved:
        subdir = ENTITY_DIR if ptype == "entity" else CONCEPT_DIR
        fp = subdir / f"{name}.md"
        if fp.exists():
            parts.append(f"### {name}（{ptype}）\n{read_page(fp)[:4000]}")
    if not parts:
        return False, "无涉及页面"
    body = "\n\n".join(parts)

    prompt = f"""【本次任务】基于以下 {len(parts)} 个相关页面，生成一张 synthesis 综合页：{strategy_topic}。

要求：
1. 「跨源综合」把多个来源的信息综合成可执行的结论。
2. 「策略建议」给出对天鹅公主的具体可落地建议（3-5 条）。
3. 「涉及实体/概念」列出所有涉及的页面。
4. 「来源列表」引用涉及的页面来源。
5. 「原文引用」从多个页面取关键句。
6. 输出纯 Markdown。

【综合主题】{strategy_topic}
【涉及页面内容】
{body}
"""
    try:
        llm = get_llm_provider()
        compiled = llm.complete(_SYSTEM_PROMPT, prompt)
    except Exception as e:
        return False, f"LLM 调用失败: {e}"
    if not compiled or not compiled.strip():
        return False, "LLM 返回空输出"

    SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    page_path = SYNTHESIS_DIR / f"{page_name}.md"
    content = (
        f"# {strategy_topic}\n\n"
        f"- 类型：synthesis\n"
        f"- 编译时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{compiled.strip()}\n"
    )
    page_path.write_text(content, encoding="utf-8")
    return True, str(page_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 comparison/synthesis 特殊页")
    ap.add_argument("--dry-run", action="store_true", help="只列出将生成的页")
    args = ap.parse_args()

    comparisons = [
        ("金粒门_vs_天鹅公主", "金粒门", "天鹅公主"),
        ("玛芝莲_vs_天鹅公主", "玛芝莲", "天鹅公主"),
    ]
    syntheses = [
        ("烘焙竞品防御策略", [("金粒门", "entity"), ("玛芝莲", "entity"), ("天鹅公主", "entity")],
         "天鹅公主烘焙竞品防御策略"),
    ]

    if args.dry_run:
        print("将生成 comparison 页:")
        for name, a, b in comparisons:
            print(f"  {name} ({a} vs {b})")
        print("将生成 synthesis 页:")
        for name, involved, topic in syntheses:
            print(f"  {name} ({topic})")
        return 0

    for name, a, b in comparisons:
        print(f"生成对比页: {name}")
        ok, res = gen_comparison(name, a, b)
        print(f"  {'✅' if ok else '❌'} {res}")

    for name, involved, topic in syntheses:
        print(f"生成综合页: {name}")
        ok, res = gen_synthesis(name, involved, topic)
        print(f"  {'✅' if ok else '❌'} {res}")

    print("comparison/synthesis 生成完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())