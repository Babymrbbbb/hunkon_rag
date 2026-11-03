"""pytest 全局初始化：在任何测试模块导入 app 之前固定环境变量。

存在的理由（2026-11-03）：
    原先各测试模块在自己的模块顶层 `os.environ[...] = ...` 后再 import app。
    但 `app.config.settings` 是模块级单例，**谁先导入谁决定配置**，
    于是结果依赖 pytest 的文件收集顺序：新增一个按字母序更靠前的测试文件
    （test_db_url.py）就先导入了 app，导致 settings 从开发者本地的 .env
    读到真实 LLM/API Key，其它测试全部 401 或外网调用失败。

    conftest.py 会在测试模块导入前执行，因此把环境固定放在这里，
    顺序依赖彻底消除，也避免误用本地 .env（.env 里可能有真实密钥）。

注意：settings 仍是进程级单例，因此所有测试共享同一个临时数据库；
各测试模块内部的 env 赋值不再改变生效配置，只作为意图说明保留。
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="swan_rag_pytest_")

# 直接赋值（不用 setdefault）：必须覆盖开发者本地 .env 的真实配置
os.environ["RAG_RAG_DB_FILE"] = os.path.join(_TMP, "test.db")
os.environ["RAG_API_KEY"] = "test-secret-key"
os.environ["RAG_LLM_PROVIDER"] = "mock"
os.environ["RAG_EMBEDDER_PROVIDER"] = "mock"
os.environ["RAG_WIKI_COMPILER"] = "rule"
os.environ["RAG_AUTO_SEED"] = "false"
os.environ["RAG_CORPUS_DIR"] = "data/corpus"
