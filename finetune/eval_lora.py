"""LoRA 领域微调 · 效果对比评估

加载「微调前（base model）」与「微调后（base + LoRA adapter）」，对同一批品牌问题生成回答，
输出对比报告，便于观察微调是否让回答更贴合知识库表述。

用法（需已训练 LoRA，或有 GPU 环境）：
    python finetune/eval_lora.py --base-model Qwen/Qwen2.5-0.5B --adapter finetune/output/lora_adapter

产物：
    finetune/output/eval_compare.md —— 微调前后回答对比报告

诚实标注：
    本脚本为评估框架；若尚未运行 train_lora.py，则只有 base 模型回答，adapter 部分标注「未训练」。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).resolve().parent / "output"

# 与 golden set 同源的品牌业务问题（覆盖核心场景）
EVAL_QUESTIONS = [
    "天鹅公主目前一共有多少家门店，主要在哪些城市？",
    "2026 年夏季天鹅公主推出了哪四款新品？",
    "冰面包是天鹅公主的在售产品吗？市场通行参考保质期是多少？",
    "竞品金粒门主打什么品类的产品？",
    "加盟天鹅公主需要满足什么条件？",
    "天鹅公主的包材用了什么材质，是否可持续？",
]


def generate(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    prompt = (
        f"<|im_start|>system\n你是天鹅公主品牌知识助手，请基于知识库内容回答。<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；未训练则跳过对比")
    args = ap.parse_args()

    # 加载 base 模型
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # 若有 adapter，加载微调后模型
    tuned_model = None
    if args.adapter and Path(args.adapter).exists():
        tuned_model = PeftModel.from_pretrained(base_model, args.adapter)

    lines = ["# LoRA 微调效果对比", ""]
    if tuned_model is None:
        lines.append("> ⚠️ 尚未加载 LoRA 适配器（未训练）。以下仅为 base 模型基线回答，微调后对比待补充。")
    else:
        lines.append("> ✅ 已加载 LoRA 适配器，以下为微调前后对比。")
    lines.append("")

    for q in EVAL_QUESTIONS:
        base_ans = generate(base_model, tokenizer, q)
        lines.append(f"## 问题：{q}")
        lines.append("")
        lines.append("**Base 模型回答：**")
        lines.append("")
        lines.append(f"> {base_ans.strip()}")
        lines.append("")
        if tuned_model is not None:
            tuned_ans = generate(tuned_model, tokenizer, q)
            lines.append("**LoRA 微调后回答：**")
            lines.append("")
            lines.append(f"> {tuned_ans.strip()}")
        else:
            lines.append("**LoRA 微调后回答：**（待训练后补充）")
        lines.append("")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "eval_compare.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 对比报告已生成: {out_path}")


if __name__ == "__main__":
    main()