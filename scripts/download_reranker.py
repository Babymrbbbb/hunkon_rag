"""下载 bge-reranker-base 的 ONNX 格式到 _models/bge-reranker-base/。

bge-reranker-base 是 278MB 的交叉编码器（XLM-RoBERTa-base），用于对 RRF 融合结果精排。
需要 ONNX 格式 + tokenizer.json。

方式一（自动）：huggingface_hub 已安装时，从 Xenova/bge-reranker-base 下载 ONNX。
方式二（手动）：给出转换指引。

用法：
  python -m scripts.download_reranker          # 自动下载（需 huggingface_hub）
  python -m scripts.download_reranker --guide  # 只打印手动转换指引
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "_models" / "bge-reranker-base"

# HuggingFace 上 Xenova 的 ONNX 导出（tokenizer.json + model_quantized.onnx）
HF_BASE = "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx"
FILES = {
    "tokenizer.json": f"{HF_BASE}/tokenizer.json",
    "model_quantized.onnx": f"{HF_BASE}/model_quantized.onnx",
}


def guide() -> None:
    print("=" * 60)
    print("手动转换 bge-reranker-base 为 ONNX 的指引")
    print("=" * 60)
    print("""
1. 安装依赖（若未安装）：
   pip install transformers torch optimum onnx onnxruntime tokenizers

2. 用 optimum 导出 ONNX：
   optimum-cli export onnx --model BAAI/bge-reranker-base --task text-classification \\
       --monolith _models/bge-reranker-base/

3. 确认产物：
   _models/bge-reranker-base/
   ├── model.onnx
   ├── tokenizer.json
   └── config.json

4. 配置环境变量：
   RAG_RERANK=local
   （模型缺失时自动降级 score，不会报错）

注意：如果你的导出的 logits 末维是词表大小（如 151669），说明是按因果 LM 导出的，
app/reranker.py 会拒收并降级。必须是 [B, 1] 或 [B, 2] 的分数头。
""")


def download() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载到: {MODEL_DIR}")
    ok = True
    for name, url in FILES.items():
        dest = MODEL_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  ⏭ 已存在: {name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        print(f"  ⬇ {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - 用户显式触发下载
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  ✅ {name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ {name} 下载失败: {e}")
            ok = False
    if ok:
        print("\n完成！配置 RAG_RERANK=local 即可启用真 cross-encoder 精排。")
        print("验证：python -c \"from app.reranker import get_reranker; print(get_reranker().name)\"")
    else:
        print("\n部分文件下载失败。可重试，或改用 --guide 手动转换。")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 bge-reranker-base ONNX 模型")
    ap.add_argument("--guide", action="store_true", help="只打印手动转换指引")
    args = ap.parse_args()

    if args.guide:
        guide()
        return 0
    return download()


if __name__ == "__main__":
    sys.exit(main())