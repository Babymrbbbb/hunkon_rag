"""LoRA 微调 · 数据准备
把 hunkon_rag 的 data/corpus/*.md（品牌知识语料）转换为「指令-回答」JSONL 训练数据。

用法：
    python finetune/prepare_data.py

产物：
    finetune/data/train.jsonl  —— 每条 {instruction, input, output}
    finetune/data/val.jsonl    —— 少量验证集

说明：corpus 是脱敏展示语料；接入本地生产知识库（RAG_CORPUS_DIR）后可换用更大语料。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
OUT_DIR = Path(__file__).resolve().parent / "data"
VAL_RATIO = 0.2
SEED = 42


def parse_md(path: Path) -> tuple[str, str] | None:
    """解析单个 Markdown：标题 → 主题，正文 → 知识内容。返回 (主题, 内容)。"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = ""
    body_lines = []
    for ln in lines:
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
        elif ln.strip():
            body_lines.append(ln.strip())
    if not title or not body_lines:
        return None
    return title, "\n".join(body_lines)


def build_examples() -> list[dict]:
    examples = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        parsed = parse_md(path)
        if not parsed:
            continue
        title, content = parsed
        # 从标题生成指令（简单模板；真实场景可用 LLM 生成更自然的指令）
        instruction = f"根据品牌知识库回答：{title}"
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": content,
        })
    return examples


def main() -> None:
    random.seed(SEED)
    examples = build_examples()
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * VAL_RATIO))
    val = examples[:n_val]
    train = examples[n_val:]

    OUT_DIR.mkdir(exist_ok=True)
    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        (OUT_DIR / name).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in data) + "\n",
            encoding="utf-8",
        )
    print(f"✓ 训练集 {len(train)} 条 → finetune/data/train.jsonl")
    print(f"✓ 验证集 {len(val)} 条 → finetune/data/val.jsonl")


if __name__ == "__main__":
    main()