"""配置分离：pydantic-settings，所有变量以 RAG_ 前缀从环境变量/.env 读取。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 数据库
    rag_db_file: str = "swan_rag.db"

    # 全局鉴权
    api_key: str = "dev-secret-key"

    # 多租户 RBAC（2026-11-04 新增）：三档角色各自的 API Key。
    # 缺省时全部回落到 api_key（兼容旧配置/测试）。
    api_key_staff: str = "dev-secret-key"
    api_key_franchise: str = ""
    api_key_ops: str = ""

    # LLM 供应商（mock | real）
    llm_provider: str = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    # 生成 token 上限。默认 1024：推荐的 sensenova-6.8-flash-lite 是推理模型，
    # 思考链会先消耗大量 token，需留足预算才能让最终答案落到 content 字段。
    llm_max_tokens: int = 1024
    # 尽力关闭推理模型的思考链（reasoning），让其把最终答案直接写入 content 字段；
    # 部分账户/模型会忽略该开关（仍走 thinking），此时依赖上面的 max_tokens 余量。
    llm_disable_reasoning: bool = True

    # 向量化供应商（mock | local | real）
    # - mock ：哈希词袋，离线确定性，CI 默认
    # - local：本地 ONNX 量化模型（CPU 推理，数据不出机）
    # - real ：OpenAI 兼容 /embeddings 接口
    embedder_provider: str = "mock"
    embedder_base_url: str = "https://api.openai.com/v1"
    embedder_api_key: str = ""
    embedder_model: str = "text-embedding-3-small"
    # mock 的向量维度（仅 mock 生效；local/real 的维度由模型决定）
    embed_dim: int = 256
    # 本地模型目录：相对路径按项目根解析；模型缺失时 local 会自动降级 mock
    embedder_local_dir: str = "_models/qwen3-embedding-0.6b"

    # RAG 参数
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 3

    # 启动自愈合：比对 corpus 哈希，库为空或源改动时自动重建索引（防「改源忘灌库」）
    auto_seed: bool = True

    # 聚合层编译器：rule=确定性规则（离线可测） | llm=真实 LLM 识别实体/概念并生成聚合页
    wiki_compiler: str = "rule"

    # 精排器：score=保持 RRF 原序（默认，零依赖） | local=本地 ONNX cross-encoder
    # 模型缺失时 local 自动降级 score（不伪造能力）
    reranker: str = "score"

    # CRAG 自纠错：0=关闭（默认） | 1=开启（LLM 判断上下文相关性，不足则改写重试）
    crag: bool = False

    # 真实语料目录：默认 data/corpus（脱敏展示）；可指向本地生产知识库（如 D:\WorkBuddy知识库\wiki）
    corpus_dir: str = "data/corpus"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore"
    )


settings = Settings()
