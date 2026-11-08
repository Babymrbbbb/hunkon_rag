"""LLM 供应商抽象层。

架构要点（ADR-001）：把 LLM 调用抽象成 LLMProvider 接口，
- MockProvider：离线确定性生成，不依赖外网，用于本地演示/评测/CICD。
- OpenAICompatibleProvider：对接任意 OpenAI 兼容网关（OpenAI / DeepSeek / 商汤网关等），
  通过环境变量注入 base_url / api_key / model，做到供应商解耦、零硬编码。
"""
from abc import ABC, abstractmethod

import httpx

from app.config import settings


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


_SYSTEM_MARK = "用户问题："
_DOC_MARK = "（来自《"


def _parse_user(user: str):
    """从 rag 拼装的 prompt 中解析出 问题 与 首个召回片段。"""
    question = ""
    first_title = ""
    first_text = ""
    if _SYSTEM_MARK in user:
        question = user.split(_SYSTEM_MARK, 1)[1].split("\n", 1)[0].strip()
    # 第一个资料块：[资料1]（来自《标题》）\n文本
    idx = user.find("[资料1]")
    if idx != -1:
        block = user[idx:]
        if _DOC_MARK in block:
            first_title = block.split(_DOC_MARK, 1)[1].split("》", 1)[0].strip()
        # 文本到下一个资料块或结尾
        body = block.split("\n", 1)[1] if "\n" in block else ""
        if "[资料2]" in body:
            body = body.split("[资料2]", 1)[0]
        first_text = body.strip()
    return question, first_title, first_text


class MockProvider(LLMProvider):
    """离线模拟生成：回显问题 + 首个召回片段，证明 RAG 管线闭环（未调用真实 LLM）。"""

    def complete(self, system: str, user: str) -> str:
        question, title, text = _parse_user(user)
        snippet = text[:120].replace("\n", " ")
        return (
            f"【离线模拟回答 · 由 MockProvider 生成，未调用真实 LLM】\n"
            f"依据《{title}》等资料，关于「{question}」：{snippet}……\n"
            f"（以上为演示 RAG 检索增强生成链路的占位回答，接入真实 LLM 后此处为模型生成内容。）"
        )


class OpenAICompatibleProvider(LLMProvider):
    """对接 OpenAI 兼容 Chat Completions 接口。"""

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("llm_api_key 未配置，无法调用真实 LLM；请设置 RAG_LLM_API_KEY")
        # 关闭推理模型的「思考链」输出，让其把最终答案直接写入 content 字段，
        # 避免 reasoning 模型把 token 预算全花在 thinking 上导致 content 为空。
        # OpenAI 兼容网关对未知字段通常忽略，对标准模型无害。
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if settings.llm_disable_reasoning:
            payload["reasoning"] = {"enabled": False}
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # 兼容推理模型：content 可能为空，退而取 reasoning_content / reasoning
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""


def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "real":
        return OpenAICompatibleProvider()
    return MockProvider()
