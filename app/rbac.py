"""多租户 RBAC——文档级权限控制（2026-11-04 新增）。

设计：
  - 三档角色：staff（一线员工）< franchise（加盟商）< ops（管理层）
  - 每个角色对应一个 API Key（环境变量 RAG_API_KEY_STAFF/FRANCHISE/OPS）
  - 文档按角色分级（data/doc_roles.json 映射 source → 最小可见角色）
  - 检索后按当前角色过滤不可见文档的 chunk

角色继承：ops 可见全部；franchise 可见 franchise+staff；staff 仅见 staff。
"""
import json
from pathlib import Path
from typing import Optional

from app.config import settings

ROOT = Path(__file__).resolve().parent.parent
DOC_ROLES_PATH = ROOT / "data" / "doc_roles.json"

# 角色层级（值越大权限越高）
ROLE_LEVELS = {"staff": 1, "franchise": 2, "ops": 3}


def role_for_key(api_key: str) -> str:
    """根据 API Key 返回角色。未匹配返回 None（无权限）。"""
    if api_key and api_key == settings.api_key_ops:
        return "ops"
    if api_key and api_key == settings.api_key_franchise:
        return "franchise"
    if api_key and api_key == settings.api_key_staff:
        return "staff"
    if api_key and api_key == settings.api_key:
        return "staff"  # 默认 key 视为 staff（兼容旧配置）
    return ""


def load_doc_roles() -> dict:
    """读取文档角色映射。返回 {source: role}。"""
    try:
        data = json.loads(DOC_ROLES_PATH.read_text(encoding="utf-8"))
        return data.get("roles", {})
    except Exception:
        return {}


def get_doc_sources(session) -> dict:
    """从数据库构建 {document_id: source} 映射。"""
    from sqlmodel import select

    from app.models import Document

    try:
        rows = session.exec(select(Document)).all()
        return {d.id: d.source for d in rows}
    except Exception:
        return {}


def invisible_sources(role: str) -> Optional[set]:
    """返回当前角色【不可见】的文档 source 集合。
    - 无效角色返回 None（调用方应视为无权限）
    - 有效角色返回「角色层级高于当前角色」的文档集合
    未在 doc_roles 中标记的文档默认 staff 可见（即对任何有效角色可见）。
    """
    if role not in ROLE_LEVELS:
        return None
    level = ROLE_LEVELS[role]
    doc_roles = load_doc_roles()
    return {source for source, doc_role in doc_roles.items()
            if ROLE_LEVELS.get(doc_role, 1) > level}


def visible_sources(role: str) -> Optional[set]:
    """返回角色可见的文档 source 集合（用于展示/调试）。
    语义：所有文档 - 不可见文档。未标记文档视为可见。
    """
    if role not in ROLE_LEVELS:
        return None
    invisible = invisible_sources(role)
    doc_roles = load_doc_roles()
    all_sources = set(doc_roles.keys())
    # 未标记文档默认可见，但它们不在 doc_roles 中；这里返回「受控文档中可见的」。
    return {s for s in all_sources if s not in invisible}


def filter_hits_by_role(hits: list, role: str, doc_sources) -> list:
    """过滤检索结果中当前角色不可见的 chunk。

    参数：
    - hits：[(score, Chunk), ...]
    - role：当前角色（staff/franchise/ops）
    - doc_sources：{document_id: source} 映射（用于判断文档角色）
    返回过滤后的 hits。
    """
    if role == "ops":
        return hits  # ops 可见全部
    if role not in ROLE_LEVELS:
        return []  # 无效角色无权限
    invisible = invisible_sources(role)
    if invisible is None:
        return []
    result = []
    for score, chunk in hits:
        source = doc_sources.get(chunk.document_id, "")
        # 未在 doc_roles 中标记的文档默认 staff 可见（即对任何有效角色可见）
        if not source or source not in invisible:
            result.append((score, chunk))
    return result