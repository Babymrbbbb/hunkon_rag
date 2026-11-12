"""LLM 批量聚合页生成——参考实现 generate_agg_pages.py 的开源复刻。

能力等价：
  - 从聚合索引读取「实体/概念 → 源文件列表」
  - 对每个页面：读取源文件 → 拼 prompt → 调 LLM 生成聚合页
  - 权威口径确定性覆盖（复用 app.rules.apply_ruling_overrides）
  - 截断检测：缺必需字段则放弃写入
  - 写入 data/wiki/entity|concept/<page>.md

与参考实现差异：
  - 参考实现直连 urllib + 环境变量配置 LLM 端点
  - hunkon_rag 复用 app.llm.get_llm_provider()（LLMProvider 抽象，配置化）
  - 参考实现读 _meta/aggregation_index.json；hunkon_rag 从 data/corpus 扫描构建聚合索引
  - 参考实现用 kb_rules.apply_ruling_overrides；hunkon_rag 用 app.rules.apply_ruling_overrides

用法：
  python -m app.agg_pages --corpus data/corpus --min-sources 2 [--dry-run] [--only 演示品牌]
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# 允许 `python -m app.agg_pages` 与 `python app/agg_pages.py` 两种启动方式
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.rules import apply_ruling_overrides, register_ruling_path  # noqa: E402

# 模式 A：登记本模块实际产出对外文本的执行路径（见 app.rules）。
register_ruling_path("agg_pages.llm_agg")

WIKI_DIR = ROOT / "data" / "wiki"
ENTITY_DIR = WIKI_DIR / "entity"
CONCEPT_DIR = WIKI_DIR / "concept"

# 品牌实体词表：这些应生成 entity 页
BRAND_ENTITIES = {
    "演示品牌", "竞品乙", "竞品甲", "黑金可颂", "联名桃酥",
    "荔枝玫瑰冻芝士", "青提冷萃欧包", "演示会员卡",
}

# 实体/概念页的必需字段
REQUIRED_FIELDS = {
    "entity": ["摘要", "关键事实", "来源"],
    "concept": ["摘要", "概念描述", "来源"],
}

_SYSTEM_PROMPT = (
    "你是企业知识库的 LLM Wiki 编译引擎。给定某个实体/概念及其多个源文件内容，"
    "生成一张聚合页。聚合页必须综合多个来源，不是单源转述。"
    "输出纯 Markdown，不要 JSON，不要解释。"
)


def fix_type(page: str, ptype: str) -> str:
    """品牌实体修正为 entity，其余保持 concept。"""
    if page in BRAND_ENTITIES:
        return "entity"
    return ptype


def scan_sources(corpus_dir: Path) -> dict:
    """扫描语料目录，构建聚合索引：{page: {"type": str, "sources": [relpath, ...]}}。

    聚合索引是参考实现 _meta/aggregation_index.json 的等价物：
    - 每个文档标题（首个 # 标题）作为一个聚合页候选
    - 正文中出现该标题词的其它文档也归入其 sources
    - 文档类型：若标题在 BRAND_ENTITIES 中则为 entity，否则 concept
    """
    docs = []
    for f in sorted(corpus_dir.rglob("*.md")):
        text = f.read_text(encoding="utf-8")
        first_line = text.splitlines()[0].strip() if text else ""
        title = first_line.lstrip("# ").strip() if first_line.startswith("#") else f.stem
        docs.append((title, f.relative_to(corpus_dir).as_posix(), text))

    agg: dict = {}
    for title, rel, text in docs:
        if title not in agg:
            ptype = "entity" if title in BRAND_ENTITIES else "concept"
            agg[title] = {"type": ptype, "sources": []}
        agg[title]["sources"].append(rel)
    # 把「正文包含其它标题词」的文档挂到对应页。
    # 用「标题变体」匹配：完整标题（如「品牌介绍：演示品牌」）很少出现在
    # 其它文档正文里，需要拆出裸品牌名/关键词变体（如「演示品牌」）才能跨引用召回。
    variants = {}
    for title in agg:
        vs = {title}
        # 品牌页变体：去掉「品牌介绍：」「竞品分析：」等前缀
        for pre in ("品牌介绍：", "竞品分析：", "产品线", "市场调研"):
            if title.startswith(pre):
                vs.add(title[len(pre):])
                break
        # 含「：」的标题，取冒号后部分作为变体
        if "：" in title:
            tail = title.split("：", 1)[1].strip()
            vs.add(tail)
            # 冒号后部分再去掉英文后缀（如「演示品牌」→「演示品牌」）
            en = re.search(r"[\u4e00-\u9fa5]+", tail)
            if en:
                vs.add(en.group(0))
            # 纯英文品牌名（如「演示品牌」）按空格拆首词作为变体
            for w in tail.split():
                if re.fullmatch(r"[A-Za-z]+", w) and len(w) >= 2:
                    vs.add(w)
        # 含「与」的标题，拆出各部分
        for part in re.split(r"[与、]", title):
            part = part.strip()
            if len(part) >= 2:
                vs.add(part)
        variants[title] = vs

    for title, rel, text in docs:
        for other in agg:
            if other == title:
                continue
            if any(v in text for v in variants[other]) and rel not in agg[other]["sources"]:
                agg[other]["sources"].append(rel)
    return agg


def read_sources(corpus_dir: Path, rel_list: list, max_srcs: int = 15, per_src: int = 1500) -> str:
    """读取多个源文件正文，返回拼接文本。"""
    picked = rel_list[:max_srcs]
    parts = []
    for rel in picked:
        fp = corpus_dir / rel
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parts.append(f"### 源 {rel}\n{text[:per_src]}")
    return "\n\n".join(parts), picked


def build_prompt(page: str, ptype: str, body: str, picked: list) -> str:
    """拼装聚合页生成 prompt。显式声明源清单，防止 LLM 编造未提供的路径。"""
    source_list = "\n".join(f"- {r}" for r in picked)
    return f"""{_SYSTEM_PROMPT}

【聚合页面名】{page}
【聚合页类型】{ptype}

【本次任务】基于以下 {len(picked)} 个源文件，生成一张聚合 {ptype} 页。

【源文件清单】（实际提供 {len(picked)} 个，只能引用以下路径，严禁编造）
{source_list}

【源文件内容】
{body}

请输出一张聚合页，必须包含以下字段：
- 摘要：100 字内综合摘要
- 关键事实 / 概念描述：跨源综合，每条标注来自哪个源
- 来源：只能列出下方实际出现的源路径
不得编造源文件之外的内容。"""


def detect_truncated(text: str, ptype: str) -> tuple:
    """截断检测：缺必需字段或末尾半截则判定截断。返回 (is_truncated, missing_fields)。

    检查的是 Markdown 小节标题（如 `## 摘要`、`**关键事实**`），而非裸词——
    裸词匹配会把「没有关键事实」这类否定句误判为已含字段（2026-11-04 测试发现）。
    """
    required = REQUIRED_FIELDS.get(ptype, REQUIRED_FIELDS["concept"])
    missing = []
    for rf in required:
        # 兼容 `## 关键事实` 与 `**关键事实**` 两种小节标题写法
        if f"## {rf}" not in text and f"**{rf}**" not in text and f"# {rf}" not in text:
            missing.append(rf)
    return bool(missing), missing


def llm_agg(page: str, ptype: str, rel_list: list, corpus_dir: Path, extra_note: str = "") -> tuple:
    """调用 LLM 生成聚合页。返回 (ok, text)。"""
    body, picked = read_sources(corpus_dir, rel_list)
    if not body.strip():
        return False, "无可用源文件内容"

    prompt = build_prompt(page, ptype, body, picked)
    if extra_note:
        prompt += f"\n\n【最高优先级指令】{extra_note}"

    try:
        llm = get_llm_provider()
        raw = llm.complete(_SYSTEM_PROMPT, prompt)
    except Exception as e:
        return False, f"LLM 调用失败: {e}"

    if not raw or not raw.strip():
        return False, "LLM 返回空输出"

    # 权威口径确定性覆盖（复用 app.rules）
    text = apply_ruling_overrides(raw, page_name=page)

    # 截断检测
    truncated, missing = detect_truncated(text, ptype)
    if truncated:
        return False, f"输出被截断（缺字段 {missing}）"

    return True, text


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 批量生成聚合页")
    ap.add_argument("--corpus", default=None, help="语料目录（默认 settings.corpus_dir）")
    ap.add_argument("--min-sources", type=int, default=2, help="最少源数量")
    ap.add_argument("--dry-run", action="store_true", help="只列出将生成的聚合页")
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定页面名")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus) if args.corpus else Path(settings.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir

    agg = scan_sources(corpus_dir)
    items = [(page, info) for page, info in agg.items() if len(info["sources"]) >= args.min_sources]
    if args.only:
        only_set = set(args.only)
        items = [(page, info) for page, info in items if page in only_set]
    items.sort(key=lambda x: -len(x[1]["sources"]))

    print(f"待生成聚合页: {len(items)} 个（min_sources={args.min_sources}）")
    for page, info in items:
        print(f"  [{info['type']}] {page}: {len(info['sources'])} 源")

    if args.dry_run:
        return 0

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    ENTITY_DIR.mkdir(exist_ok=True)
    CONCEPT_DIR.mkdir(exist_ok=True)

    generated = 0
    for i, (page, info) in enumerate(items, 1):
        ptype = fix_type(page, info["type"])
        subdir = ENTITY_DIR if ptype == "entity" else CONCEPT_DIR
        page_path = subdir / f"{page}.md"
        print(f"[{i}/{len(items)}] 生成聚合页: {page}")

        ok, compiled = llm_agg(page, ptype, info["sources"], corpus_dir)
        if not ok:
            print(f"  ❌ 失败: {compiled}")
            continue

        content = (
            f"# {page}\n\n"
            f"- 类型：{ptype}（聚合页）\n"
            f"- 编译时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"{compiled.strip()}\n"
        )
        page_path.write_text(content, encoding="utf-8")
        print(f"  ✅ 已生成: {page_path}（{len(info['sources'])} 源）")
        generated += 1

    print(f"\n聚合页生成完成: {generated} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())