"""聚合层（wiki 层）：把 raw corpus 编译为可导航的聚合页。

这是 llm-wiki「raw → wiki → RAG 双轨」的 swan_rag 可运行版。

两种编译器（通过 RAG_WIKI_COMPILER 切换）：
- rule（默认）：确定性规则聚合（业务词表 + 文档首行），离线可跑、可测试、可复现。
- llm：调用真实 LLM 识别语料中的实体/概念，并生成聚合页（摘要/关键事实/来源），
  是 llm-wiki「LLM 参与知识组织」的完整形态。

聚合页是检索的第一入口：命中聚合页 → 返回聚合内容与来源列表；
未命中则 fallback 到原始 chunk（RAG 常规路径）。

产出：
- `data/wiki/index.json`：聚合索引（entity/concept → 源文件列表）
- 内存聚合页对象：标题、摘要、关键事实、来源
"""
import json
import re
from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.llm import get_llm_provider

# 业务实体/概念词表（rule 编译器用）：与 business_facts/store_facts 一致
_ENTITY_TERMS = {
    "黑金可颂": "entity/黑金可颂",
    "联名桃酥": "entity/联名桃酥",
    "荔枝玫瑰冻芝士": "entity/荔枝玫瑰冻芝士",
    "青提冷萃欧包": "entity/青提冷萃欧包",
    "冰面包": "entity/冰面包",
    "金粒门": "entity/金粒门",
    "玛芝莲": "entity/玛芝莲",
    "天鹅公主": "entity/天鹅公主",
    "天鹅卡": "entity/天鹅卡",
}

_CONCEPT_TERMS = {
    "门店网络": "concept/门店网络",
    "加盟政策": "concept/加盟政策",
    "包材设计": "concept/包材设计",
    "会员体系": "concept/会员体系",
    "供应链": "concept/供应链",
    "数字化": "concept/数字化",
    "运营FAQ": "concept/运营FAQ",
    "夏季新品": "concept/夏季新品",
}

WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"
INDEX_PATH = WIKI_DIR / "index.json"

_LLM_COMPILE_SYSTEM = (
    "你是企业知识库的编译引擎。给定多篇品牌资料，请：\n"
    "1. 识别其中出现的实体（品牌/产品/人物/公司名）与概念（业务主题/政策/流程）；\n"
    "2. 对每个实体/概念，输出聚合页，字段为 JSON。\n"
    "输出必须是合法 JSON 数组，每个元素：\n"
    '{"title":"实体或概念名","kind":"entity 或 concept","summary":"跨来源综合摘要(80字内)",'
    '"facts":["关键事实1","关键事实2"],"sources":["来源文档标题..."]}\n'
    "只输出 JSON，不要任何额外文字。"
)


class WikiPage:
    """聚合页：标题、摘要、关键事实、来源文档列表。"""

    def __init__(self, title: str, kind: str, summary: str, facts: List[str], sources: List[str]):
        self.title = title
        self.kind = kind  # entity | concept
        self.summary = summary
        self.facts = facts
        self.sources = sources

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
            "facts": self.facts,
            "sources": self.sources,
        }


def _clean_title(text: str) -> str:
    """去掉 markdown 标题符号。"""
    return text.lstrip("# ").strip()


def _extract_key_facts(text: str) -> List[str]:
    """从正文提取关键事实：包含数字/日期/温度等的句子（简单规则）。"""
    sentences = re.split(r"[。\n]+", text)
    facts = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if re.search(r"\d|℃|%|㎡|万元|元|周|天", s):
            facts.append(s)
    return facts[:5]


def _load_docs(corpus_dir: Path) -> List[tuple[str, str]]:
    """读取 corpus 全部文档 (title, content)。"""
    docs = []
    for f in sorted(corpus_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        title = _clean_title(text.splitlines()[0]) if text.startswith("#") else f.stem
        docs.append((title, text))
    return docs


class WikiIndex:
    """聚合层（rule 编译器）：按业务词表确定性聚合。"""

    def __init__(self, corpus_dir: Path, generated_by: str = "deterministic-rule-based"):
        self.corpus_dir = corpus_dir
        self.pages: Dict[str, WikiPage] = {}
        self.term_to_page: Dict[str, str] = {}  # 业务词 -> 聚合页标题
        self.generated_by = generated_by
        self._build()

    def _build(self) -> None:
        docs = _load_docs(self.corpus_dir)
        for term, page_path in {**_ENTITY_TERMS, **_CONCEPT_TERMS}.items():
            sources = [t for t, c in docs if term in c]
            if not sources:
                continue
            summary = ""
            facts: List[str] = []
            for t, c in docs:
                if term in c:
                    summary = c.split("\n", 1)[-1].strip()[:120]
                    facts = _extract_key_facts(c)
                    break
            page_title = page_path.split("/", 1)[1]
            self.pages[page_title] = WikiPage(
                title=page_title,
                kind=page_path.split("/", 1)[0],
                summary=summary,
                facts=facts,
                sources=sources,
            )
            self.term_to_page[term] = page_title

    def search(self, query: str) -> List[WikiPage]:
        """检索聚合页：命中业务词则返回对应聚合页。

        排序策略（2026-11-03 审查修复）：品牌名（如「天鹅公主」）会出现在绝大多数
        业务问题里，若按词表遍历顺序返回，会把「天鹅公主」万能页顶到最前，掩盖
        真正命中的实体/概念页。这里把命中词按「非品牌名优先」排序：
        - 具体实体/概念（冰面包、门店网络、加盟政策…）优先；
        - 品牌名聚合页作为兜底放最后（仅在无更具体命中时返回）。
        """
        BRAND_TERMS = {"天鹅公主"}
        hits = []
        for term, page_title in self.term_to_page.items():
            if term in query:
                hits.append((term, self.pages[page_title]))
        # 有具体实体/概念命中时，剔除品牌兜底页（否则每条品牌问题都会被注入通用品牌简介）
        specific = [h for h in hits if h[0] not in BRAND_TERMS]
        if specific:
            return [p for _term, p in specific]
        return [p for _term, p in hits]

    def to_index_json(self) -> dict:
        return {
            "generated": self.generated_by,
            "pages": {title: p.to_dict() for title, p in self.pages.items()},
        }

    def save_index(self, force: bool = False) -> None:
        """把聚合索引落盘。

        并发压测优化（2026-10-30）：/chat 每次请求都会调用 save_index()。
        若索引内容未变化，跳过写盘，避免高并发下大量进程内线程同时写同一 JSON 文件。
        仅当聚合层实际重建（force=True，构建时调用）才落盘。
        """
        if not force:
            return
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps(self.to_index_json(), ensure_ascii=False, indent=2), encoding="utf-8")


class LLMWikiIndex(WikiIndex):
    """聚合层（llm 编译器）：调用真实 LLM 识别实体/概念并生成聚合页。

    流程：
    1. 把 corpus 文档（标题 + 前 N 字符）拼进提示词；
    2. 调用真实 LLM（sensenova 等），要求输出 JSON 数组；
    3. 解析 JSON 生成 WikiPage 列表，并建立 term->page 映射。
    """

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.pages: Dict[str, WikiPage] = {}
        self.term_to_page: Dict[str, str] = {}
        self.generated_by = "llm-compiler"
        self._build_llm()

    def _build_llm(self) -> None:
        docs = _load_docs(self.corpus_dir)
        # 拼资料：标题 + 正文前 300 字（控制 token）
        corpus_block = "\n\n".join(
            f"【{title}】\n{content[:300]}" for title, content in docs
        )
        user_msg = f"以下是品牌资料：\n\n{corpus_block}\n\n请识别实体/概念并输出聚合页 JSON。"

        llm = get_llm_provider()
        raw = llm.complete(_LLM_COMPILE_SYSTEM, user_msg)
        pages = self._parse_llm_output(raw)
        for page in pages:
            self.pages[page.title] = page
            self.term_to_page[page.title] = page.title

    def _parse_llm_output(self, raw: str) -> List[WikiPage]:
        """解析 LLM 输出的 JSON（容错：剥代码块/取第一个 JSON 数组）。"""
        text = raw.strip()
        # 去掉可能的 ```json ... ``` 包裹
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        # 提取第一个 [ ... ] 数组
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            # 降级：用规则词表兜底
            return self._fallback_rule_pages()
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return self._fallback_rule_pages()
        pages = []
        for item in data:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            kind = str(item.get("kind", "concept")).strip()
            summary = str(item.get("summary", "")).strip()
            facts = [str(f) for f in item.get("facts", []) if str(f).strip()]
            sources = [str(s) for s in item.get("sources", []) if str(s).strip()]
            if not title or not summary:
                continue
            pages.append(WikiPage(title=title, kind=kind, summary=summary, facts=facts, sources=sources))
        if not pages:
            return self._fallback_rule_pages()
        return pages

    def _fallback_rule_pages(self) -> List[WikiPage]:
        """LLM 输出解析失败时，用规则词表生成聚合页兜底。"""
        rule_index = WikiIndex(self.corpus_dir)
        return list(rule_index.pages.values())


# 进程级缓存：同一份语料只构建一次聚合层（2026-10-30 压测优化）。
# 用语料哈希判断是否变化；变化才重建。避免每次 /chat 都重复构建 wiki 索引。
_wiki_cache: dict = {"hash": None, "index": None, "dir_mtime": None}


def _corpus_digest(corpus_dir: Path) -> str:
    """计算语料哈希（与 seed.corpus_hash 相同的遍历方式）。

    并发压测优化（2026-10-30）：在缓存命中场景，每次 /chat 都全盘 rglob + 读全部
    *.md 算哈希是纯浪费（语料没变哈希不会变）。因此先比较目录 mtime 快照：
    目录未变则直接复用上一次哈希值，避免每次请求都做磁盘 IO。
    """
    import hashlib
    # 目录最新 mtime 快照（含子目录文件）；若与上次一致则复用缓存哈希
    latest_mtime = 0.0
    for f in corpus_dir.rglob("*.md"):
        try:
            m = f.stat().st_mtime
        except OSError:
            continue
        if m > latest_mtime:
            latest_mtime = m
    if latest_mtime == _wiki_cache.get("dir_mtime") and _wiki_cache.get("hash") is not None:
        return _wiki_cache["hash"]
    h = hashlib.sha256()
    for f in sorted(corpus_dir.rglob("*.md")):
        h.update(f.read_bytes())
    digest = h.hexdigest()
    _wiki_cache["dir_mtime"] = latest_mtime
    _wiki_cache["hash"] = digest
    return digest


def build_wiki(corpus_dir: Path = None, save: bool = True) -> WikiIndex:
    """构建聚合层。按配置选择编译器：llm 或 rule。带进程级缓存。

    并发压测优化（2026-10-30）：
    - 每次调用都全盘扫描算哈希是浪费（语料没变哈希不会变）。缓存命中直接返回。
    - 缓存命中时不需要写盘（索引没有变化）；仅构建后（语料确实变化）落盘。
    - save=False 供只读路径（如 /chat）调用，跳过 save_index 写盘。
    """
    if corpus_dir is None:
        corpus_dir = Path(settings.corpus_dir)
    # 语料哈希（带目录 mtime 快照缓存，避免每次请求全盘扫描）
    digest = _corpus_digest(corpus_dir)
    if _wiki_cache["hash"] == digest and _wiki_cache["index"] is not None:
        return _wiki_cache["index"]
    if settings.wiki_compiler == "llm":
        index = LLMWikiIndex(corpus_dir)
    else:
        index = WikiIndex(corpus_dir)
    _wiki_cache["hash"] = digest
    _wiki_cache["index"] = index
    if save:
        index.save_index(force=True)
    return index