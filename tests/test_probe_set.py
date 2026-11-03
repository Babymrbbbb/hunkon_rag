"""语义探针集与检索缓存的回归测试。

两件事：
1. 探针集必须指向真实存在的文档标题，否则「未召回」无法区分是检索差还是题写错。
2. 检索缓存必须随「向量化实现」失效：同语料重建时 chunk 行数与最大 id 不变，
   只靠行数指纹会命中旧缓存，表现为「换模型后效果不升反降」（2026-11-03 实测踩到）。
"""
import json
from pathlib import Path

import pytest
from sqlmodel import Session

from app import retriever as R
from app.config import settings
from app.database import engine, init_db
from app.seed import force_rebuild

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = ROOT / "data" / "eval_semantic_probe.json"


@pytest.fixture(autouse=True)
def _ensure_schema():
    """单独跑本文件时数据库可能尚未建表（其他用例由各自流程触发）。"""
    init_db()


def _probes():
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))


def _corpus_titles():
    titles = set()
    for f in sorted((ROOT / "data" / "corpus").glob("*.md")):
        first = f.read_text(encoding="utf-8").splitlines()[0]
        titles.add(first.lstrip("# ").strip())
    return titles


def test_probe_entries_target_existing_docs():
    titles = _corpus_titles()
    missing = [p["expected_doc"] for p in _probes() if p["expected_doc"] not in titles]
    assert not missing, f"探针集指向了不存在的文档：{missing}"


def test_probe_set_has_enough_samples():
    """样本量下限：12 条时测出的百分比不可信（2026-11-03 实测，75% 扩样后回落到 61.3%）。"""
    n = len(_probes())
    assert n >= 30, f"探针集只有 {n} 条，低于 30 条时百分比波动过大，不足以对外引用"


def test_every_corpus_doc_is_probed():
    """每篇语料至少被问一次，否则该文档的检索质量无人观测。"""
    probed = {p["expected_doc"] for p in _probes()}
    uncovered = sorted(_corpus_titles() - probed)
    assert not uncovered, f"以下文档没有任何探针覆盖，质量盲区：{uncovered}"


def test_probe_questions_are_unique_and_documented():
    probes = _probes()
    questions = [p["question"] for p in probes]
    assert len(questions) == len(set(questions)), "探针问题存在重复"
    for p in probes:
        assert len(p["question"]) >= 6, f"问题过短，可能是误填：{p['question']}"
        assert p.get("why_semantic"), f"缺少语义改写说明：{p['question']}"


def test_probe_set_stays_out_of_the_gate():
    """探针集不参与门禁：golden 与 probe 必须是两个文件，防止有人把语义题塞进 CI 门禁。"""
    golden = json.loads((ROOT / "data" / "eval_golden.json").read_text(encoding="utf-8"))
    golden_q = {g["question"] for g in golden}
    overlap = golden_q & {p["question"] for p in _probes()}
    assert not overlap, f"探针题混入了门禁用例：{overlap}"


def test_corpus_fingerprint_tracks_embedder():
    """语料与数据行数完全不变时，仅换向量维度也必须产生不同指纹。"""
    with Session(engine) as s:
        force_rebuild(s)
        original_dim = settings.embed_dim
        try:
            settings.embed_dim = 64
            fp_a = R._corpus_hash_fingerprint(s)
            settings.embed_dim = 96
            fp_b = R._corpus_hash_fingerprint(s)
        finally:
            settings.embed_dim = original_dim
    assert fp_a != fp_b, "指纹未随向量化实现变化——换模型后会命中旧缓存"


def test_rebuild_drops_retrieval_cache():
    """重建索引后缓存必须置空，下一次检索重新装载。"""
    with Session(engine) as s:
        force_rebuild(s)
        R.get_cached_chunks(s)
        assert R._CHUNK_CACHE["chunks"] is not None

        force_rebuild(s)
        assert R._CHUNK_CACHE["chunks"] is None, "重建后缓存未失效，检索会继续用旧向量"


def test_probe_report_runs_offline(monkeypatch):
    """探针脚本在 mock 档下可跑通且退出码为 0（它是观测工具，不是门禁）。"""
    from app import eval_probe

    monkeypatch.setattr("sys.argv", ["eval_probe", "--embedders", "mock", "--json"])
    assert eval_probe.main() == 0
