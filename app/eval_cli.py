"""评测 CI/CD 命令行工具（LLMOps 雏形）。

用法：
    python -m app.eval_cli                # 跑全部 golden 用例，阈值 0.75，生成报告
    python -m app.eval_cli --threshold 0.8   # 自定义阈值
    python -m app.eval_cli --json        # 输出 JSON 到 stdout（供 CI 解析）
    python -m app.eval_cli --mock         # 强制离线 MockProvider + rule 编译（CI 可复现）

行为：
    1. 跑 golden set（复用 app.eval.run_eval）
    2. 生成 reports/eval_report_<timestamp>.md（Markdown 报告）
    3. 追加 reports/history.json（时间序列，供监控看板使用）
    4. 召回命中率 < threshold 时以非零退出码退出（失败拦截，CI 可据此阻断）

--mock 说明：
    生产 .env 可能配置了真实 LLM（RAG_LLM_PROVIDER=real + RAG_WIKI_COMPILER=llm），
    评测会撞上游限流。--mock 在 import 项目配置前强制设置环境变量，让评测
    使用离线 MockProvider + 确定性 rule 编译——保证 CI 可复现、不受外部限流影响。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# --mock 必须在 import app.settings 之前设置环境变量（配置在模块导入时读取）
# 因此这里不能像常规那样在 main() 里处理参数后再设置，需要在模块顶部先解析一次。
_pre_ap = argparse.ArgumentParser(add_help=False)
_pre_ap.add_argument("--mock", action="store_true")
_pre_args, _ = _pre_ap.parse_known_args()
if _pre_args.mock:
    os.environ["RAG_LLM_PROVIDER"] = "mock"
    os.environ["RAG_WIKI_COMPILER"] = "rule"
    os.environ["RAG_EMBEDDER_PROVIDER"] = "mock"

from sqlmodel import Session

from app.database import engine, init_db
from app.eval import run_eval
from app.seed import reseed_if_needed

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _load_history() -> list[dict]:
    history_path = REPORTS_DIR / "history.json"
    if history_path.exists():
        return json.loads(history_path.read_text(encoding="utf-8"))
    return []


def _save_history(history: list[dict]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="swan_rag 评测门禁（LLMOps CI）")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="召回命中率阈值，低于则非零退出（默认 0.75）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 到 stdout")
    ap.add_argument("--mock", action="store_true",
                    help="强制离线 MockProvider + rule 编译（已在模块顶部生效）")
    args = ap.parse_args()

    init_db()
    with Session(engine) as session:
        # CI 是全新环境，数据库为空时检索不到任何文档 → 召回率恒为 0 导致误报失败。
        # 评测前先走自愈合灌库（源与库一致时零开销跳过），保证 CI 可复现。
        reseed_if_needed(session)
        report = run_eval(session)

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "total": report.total,
        "retrieval_hit_rate": report.retrieval_hit_rate,
        "faithfulness_rate": report.faithfulness_rate,
        "threshold": args.threshold,
        "passed": report.retrieval_hit_rate >= args.threshold,
    }

    # 写时间序列历史
    history = _load_history()
    history.append(record)
    _save_history(history)

    # 生成 Markdown 报告
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"eval_report_{ts}.md"
    lines = [
        f"# swan_rag 评测报告 · {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 用例总数: {report.total}",
        f"- 召回命中率: **{report.retrieval_hit_rate:.1%}**（阈值 {args.threshold:.0%}）",
        f"- 忠实度: **{report.faithfulness_rate:.1%}**",
        f"- 结果: {'✅ 通过' if record['passed'] else '❌ 未达阈值'}",
        "",
        "## 逐题明细",
        "",
        "| 问题 | 期望文档 | Top1 召回 | 召回命中 | 忠实命中 |",
        "|---|---|---|---|---|",
    ]
    for d in report.details:
        lines.append(
            f"| {d.question[:40]} | {d.expected_doc} | {d.retrieved_top1_doc} "
            f"| {'✅' if d.retrieval_hit else '❌'} | {'✅' if d.faithfulness_hit else '❌'} |"
        )
    lines.append("")
    lines.append(f"> {report.note}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        status = "✅ 通过" if record["passed"] else "❌ 未达阈值（非零退出）"
        print(f"评测完成: 召回 {report.retrieval_hit_rate:.1%} / 忠实 {report.faithfulness_rate:.1%} · {status}")
        print(f"报告: {report_path}")

    # 失败拦截：未达阈值返回非零退出码（CI 可据此阻断）
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())