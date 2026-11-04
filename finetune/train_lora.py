"""LoRA 领域微调 · 训练脚本（Qwen2.5-0.5B + peft + transformers）

在 hunkon_rag 品牌知识语料上做领域微调，让模型回答品牌业务问题时更贴合知识库表述。

用法（需 GPU 环境，或 CPU 但训练时间很长）：
    pip install torch transformers peft datasets accelerate
    python finetune/train_lora.py --model Qwen/Qwen2.5-0.5B --epochs 3

产物：
    finetune/output/lora_adapter/  —— LoRA 适配器权重（可直接用 peft 加载）
    finetune/output/train_loss.json —— 训练 loss 记录

诚实标注：
    本脚本为可复现实验设计；在本项目编写环境（无 GPU 的 Windows 本机）未实际执行训练。
    在任意有 GPU 的环境按 README 即可运行。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "output"


def format_example(example: dict) -> str:
    """把指令数据组装成 Qwen 聊天格式。"""
    return (
        f"<|im_start|>system\n你是Hunkon 品牌知识助手，请基于知识库内容回答。<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"
    )


def tokenize(example: dict, tokenizer):
    text = format_example(example)
    enc = tokenizer(text, truncation=True, max_length=512, padding=False)
    enc["labels"] = enc["input_ids"].copy()
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()

    # 加载 tokenizer 与模型（4bit 量化可显著降显存，可选）
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置：只训练低秩适配器，冻结原模型
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    train_ds = load_dataset(
        "json", data_files=str(DATA_DIR / "train.jsonl"), split="train"
    ).map(lambda x: tokenize(x, tokenizer), batched=False)
    val_ds = load_dataset(
        "json", data_files=str(DATA_DIR / "val.jsonl"), split="train"
    ).map(lambda x: tokenize(x, tokenizer), batched=False)

    # 训练参数
    out_dir = OUT_DIR / "lora_adapter"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    # 训练
    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 保存 loss 记录
    loss_history = [{"step": s, "loss": l} for s, l in enumerate(train_result.training_loss, 1)]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "train_loss.json").write_text(
        json.dumps(loss_history, indent=2), encoding="utf-8"
    )
    print(f"✓ LoRA 适配器已保存: {out_dir}")
    print(f"✓ 训练 loss 记录: {OUT_DIR / 'train_loss.json'}")


if __name__ == "__main__":
    main()