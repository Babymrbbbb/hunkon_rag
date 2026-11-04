"""多租户 RBAC 测试：角色识别、文档分级、可见性继承、检索过滤。

背景：
  2026-11-04 新增多租户权限：staff/franchise/ops 三档，文档按角色分级，
  检索后过滤不可见文档。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from app.config import settings  # noqa: E402
from app.rbac import (  # noqa: E402
    ROLE_LEVELS,
    filter_hits_by_role,
    load_doc_roles,
    role_for_key,
    visible_sources,
)


class _FakeChunk:
    def __init__(self, cid):
        self.id = cid
        self.document_id = cid


def test_role_levels():
    assert ROLE_LEVELS == {"staff": 1, "franchise": 2, "ops": 3}


def test_role_for_key_default_staff():
    """默认 key（settings.api_key）→ staff。"""
    assert role_for_key(settings.api_key) == "staff"


def test_role_for_key_custom():
    """配置了三档 key 后正确识别。"""
    old_staff = settings.api_key_staff
    old_fr = settings.api_key_franchise
    old_ops = settings.api_key_ops
    settings.api_key_staff = "staff-key"
    settings.api_key_franchise = "franchise-key"
    settings.api_key_ops = "ops-key"
    try:
        assert role_for_key("staff-key") == "staff"
        assert role_for_key("franchise-key") == "franchise"
        assert role_for_key("ops-key") == "ops"
        assert role_for_key("unknown") == ""
    finally:
        settings.api_key_staff = old_staff
        settings.api_key_franchise = old_fr
        settings.api_key_ops = old_ops


def test_load_doc_roles():
    roles = load_doc_roles()
    assert "10_franchise.md" in roles
    assert roles["10_franchise.md"] == "franchise"
    assert "06_supplychain.md" in roles
    assert roles["06_supplychain.md"] == "ops"


def test_visible_sources_inheritance():
    """ops 可见全部；franchise 可见 franchise+staff；staff 仅见 staff。"""
    old_staff = settings.api_key_staff
    old_fr = settings.api_key_franchise
    old_ops = settings.api_key_ops
    settings.api_key_staff = "s"
    settings.api_key_franchise = "f"
    settings.api_key_ops = "o"
    try:
        staff_vis = visible_sources("staff")
        fr_vis = visible_sources("franchise")
        ops_vis = visible_sources("ops")
        # staff 看不到 franchise/ops 文档
        assert "10_franchise.md" not in staff_vis
        assert "06_supplychain.md" not in staff_vis
        # franchise 能看到 franchise 文档，但看不到 ops
        assert "10_franchise.md" in fr_vis
        assert "06_supplychain.md" not in fr_vis
        # ops 全可见
        assert "10_franchise.md" in ops_vis
        assert "06_supplychain.md" in ops_vis
    finally:
        settings.api_key_staff = old_staff
        settings.api_key_franchise = old_fr
        settings.api_key_ops = old_ops


def test_filter_hits_by_role():
    """过滤逻辑：ops 全返回，staff 剔除受限文档，无效角色返回空。"""
    hits = [
        (0.9, _FakeChunk(1)),
        (0.8, _FakeChunk(2)),
        (0.7, _FakeChunk(3)),
    ]
    # doc_sources: 1=普通文档(staff可见), 2=加盟政策(franchise), 3=供应链(ops)
    doc_sources = {1: "00_brand.md", 2: "10_franchise.md", 3: "06_supplychain.md"}

    # ops 全可见
    ops = filter_hits_by_role(hits, "ops", doc_sources)
    assert len(ops) == 3

    # staff 仅见 doc1
    staff = filter_hits_by_role(hits, "staff", doc_sources)
    assert [c.id for _s, c in staff] == [1]

    # franchise 见 doc1+doc2
    fr = filter_hits_by_role(hits, "franchise", doc_sources)
    assert sorted(c.id for _s, c in fr) == [1, 2]

    # 无效角色返回空
    assert filter_hits_by_role(hits, "", doc_sources) == []


def test_filter_hits_unrestricted_docs_visible_to_all():
    """未在 doc_roles 标记的文档默认对任何有效角色可见。"""
    hits = [(0.9, _FakeChunk(99))]
    doc_sources = {99: "99_unknown.md"}
    staff = filter_hits_by_role(hits, "staff", doc_sources)
    assert len(staff) == 1

# ---------- API 集成测试 ----------

def test_search_filters_by_role():
    """staff key 搜索不到 ops 文档；ops key 可以。"""
    from fastapi.testclient import TestClient
    from sqlmodel import Session

    from app.database import engine, init_db
    from app.main import app
    from app.rag import ingest

    init_db()
    with Session(engine) as s:
        # 灌入两篇文档：普通 + ops
        ingest(s, title="品牌介绍", content="天鹅公主是清远品牌", source="00_brand.md")
        ingest(s, title="供应链", content="报损率控制在 3% 以内", source="06_supplychain.md")

    c = TestClient(app)

    # staff key（默认 test-secret-key）
    r = c.post("/search", json={"query": "报损率", "top_k": 5}, headers={"X-API-Key": "test-secret-key"})
    titles = [h["title"] for h in r.json().get("hits", [])]
    assert "供应链" not in titles, f"staff 不应看到供应链文档: {titles}"

    # ops key（临时设置）
    old_ops = settings.api_key_ops
    settings.api_key_ops = "ops-secret"
    try:
        r2 = c.post("/search", json={"query": "报损率", "top_k": 5}, headers={"X-API-Key": "ops-secret"})
        titles2 = [h["title"] for h in r2.json().get("hits", [])]
        assert "供应链" in titles2, f"ops 应看到供应链文档: {titles2}"
    finally:
        settings.api_key_ops = old_ops
