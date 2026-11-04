"""语义探针评测（**非 CI 门禁**）。

用途：单独量化「向量层的语义泛化能力」，与 eval_cli 的召回门禁解耦。

为什么要单独一套：`data/eval_golden.json` 那 8 条偏关键词型（问题里直接出现品牌名、
产品名），BM25 关键词路本就能命中，RRF 融合会把向量质量的差异掩盖掉——实测 mock
与 local 都是 8/8，看不出真假向量的区别。本探针集改用「同义改写、与原文字面几乎
不重叠」的问法（如「市面上那种冰冰凉的面包，我们店里有没有在卖？」→《冰面包市场调研》的
「非在售」结论），才能区分词袋哈希与真语义向量。

退出码恒为 0（仅参数错误除外）：这是质量观测工具，不参与门禁——否则 mock 环境
（CI 无 613 MB 模型）会每次都判失败。

用法：
    python -m app.eval_probe                        # mock vs local 对照
    python -m app.eval_probe --embedders mock       # 只跑 mock
    python -m app.eval_probe --embedders mock,local --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from app.config import settings
from app.database import engine, init_db
from app.embedder import get_embedder
from app.models import Document
from app.retriever import Retriever
from app.seed import force_rebuild

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = ROOT / "data" / "eval_semantic_probe.json"
REPORT_DIR = ROOT / "reports"
TOP_K = 3


def load_probes() -> list:
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))


def run_one(session: Session, provider: str, probes: list) -> dict:
    """用指定向量 provider 重建索引后逐题检索。"""
    # settings 是模块级单例，get_embedder() 按 provider 名缓存实例，切换即时生效。
    settings.embedder_provider = provider
    emb = get_embedder()
    force_rebuild(session)  # 维度随 provider 变化，必须重建，否则新旧向量不可比

    fingerprint = emb.fingerprint
    degraded = provider == "local" and fingerprint.startswith("mock:")
    ret = Retriever(emb, TOP_K)

    rows = []
    for case in probes:
        q = case["question"]
        expected = case["expected_doc"]
        hits = ret.search(session, emb.embed_query(q), TOP_K, query_text=q, use_hybrid=True)
        titles = []
        for _score, ch in hits:
            doc = session.get(Document, ch.document_id)
            titles.append(doc.title if doc else "?")
        rows.append(
            {
                "question": q,
                "expected_doc": expected,
                "retrieved": titles,
                "top1_doc": titles[0] if titles else "",
                "top1_hit": bool(titles) and titles[0] == expected,
                "top3_hit": expected in titles,
            }
        )

    total = len(rows) or 1
    return {
        "provider": provider,
        "fingerprint": fingerprint,
        "degraded": degraded,
        "total": len(rows),
        "top1_rate": round(sum(r["top1_hit"] for r in rows) / total, 3),
        "top3_rate": round(sum(r["top3_hit"] for r in rows) / total, 3),
        "rows": rows,
    }


def to_markdown(results: list, now: datetime) -> str:
    lines = [
        f"# 语义探针评测 · {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> 探针集：`data/eval_semantic_probe.json`（同义改写、与原文字面几乎不重叠）。",
        "> 本报告是**质量观测**，不参与 CI 门禁；退出码恒为 0。",
        "",
        "## 一、汇总",
        "",
        "| provider | 指纹 | top1 命中率 | top3 命中率 | 降级 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['provider']} | `{r['fingerprint']}` | {r['top1_rate']:.1%} "
            f"| {r['top3_rate']:.1%} | {'⚠️ 模型缺失已降级' if r['degraded'] else '否'} |"
        )

    lines += ["", "## 二、逐题对照", ""]
    header = "| 探针问题 | 期望文档 |" + "".join(f" {r['provider']} top1 |" for r in results)
    sep = "|---|---|" + "---|" * len(results)
    lines += [header, sep]
    for i in range(len(results[0]["rows"])):
        q = results[0]["rows"][i]["question"]
        exp = results[0]["rows"][i]["expected_doc"]
        cells = []
        for r in results:
            row = r["rows"][i]
            if row["top1_hit"]:
                cells.append(" ✅ 命中 |")
            elif row["top3_hit"]:
                cells.append(f" ⚠️ 仅 top3（{row['top1_doc']}） |")
            else:
                cells.append(f" ❌ 未召回（{row['top1_doc']}） |")
        lines.append(f"| {q} | {exp} |" + "".join(cells))

    lines += [
        "",
        "## 三、怎么读这份报告",
        "",
        "- **top1 命中**：检索第一条就是正确答案，生成阶段上下文最干净。",
        "- **仅 top3**：答案在上下文中但排位靠后，生成阶段仍可答对，只是噪音更多。",
        "- **未召回**：golden 意义上的失败，模型只能答「资料未提及」。",
        "- mock 是词袋哈希，只能靠字面重叠；local 是 qwen3 真语义向量。两者差距即为"
        "「真向量在本语料上的收益」。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="语义探针评测（非门禁，退出码恒 0）")
    ap.add_argument("--embedders", default="mock,local", help="逗号分隔，如 mock,local")
    ap.add_argument("--json", action="store_true", help="只输出 JSON（不写 Markdown 报告）")
    args = ap.parse_args()

    providers = [p.strip() for p in args.embedders.split(",") if p.strip()]
    probes = load_probes()

    init_db()
    results = []
    with Session(engine) as session:
        for provider in providers:
            results.append(run_one(session, provider, probes))

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    now = datetime.now()
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"probe_{now.strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(to_markdown(results, now), encoding="utf-8")

    for r in results:
        flag = " (降级)" if r["degraded"] else ""
        print(
            f"{r['provider']:<6} fp={r['fingerprint']:<12} "
            f"top1={r['top1_rate']:.1%} top3={r['top3_rate']:.1%}{flag}"
        )
    print(f"\n报告: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
