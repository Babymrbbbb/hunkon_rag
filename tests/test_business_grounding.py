"""业务事实接地护栏：确保 data/business_facts.json 中每条业务事实在 corpus 中可检索到。

与 test_grounding.py（门店数据一致性）互补：本测试负责「门店以外」的关键业务数字/日期，
要求每条事实的 must_appear 关键词都出现在 data/corpus/*.md 中。作用：
- 新增/修改业务事实后若未同步到 corpus，pytest 失败，阻断错误资料上线；
- corpus 被改成与真相源矛盾（漏改/错改），pytest 失败，防止资料漂移。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
FACTS_PATH = ROOT / "data" / "business_facts.json"

facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def _corpus_text() -> str:
    return "\n".join(
        f.read_text(encoding="utf-8") for f in sorted(CORPUS_DIR.glob("*.md"))
    )


def test_every_claim_must_appear_in_corpus():
    text = _corpus_text()
    missing = []
    for key, claim in facts["claims"].items():
        for kw in claim["must_appear"]:
            if kw not in text:
                missing.append(f"{key}: 缺少关键词『{kw}』")
    assert not missing, (
        "以下业务事实未在 corpus 中出现（真相源已更新但 corpus 未同步？）:\n"
        + "\n".join(f"  - {m}" for m in missing)
    )