"""Offline contract and evaluation tests for Phase 6 runner, hash gate, resume, and tables."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from cert_recovery.adjudication.domain import InsuranceClaim, rule_decision
from cert_recovery.adjudication.phase4_runner import FakeRoleSpecialistClient
from cert_recovery.oracle.phase6_eval import (
    _load_jsonl,
    _single_agent_prompt,
    rebuild_phase6_tables,
    run_live_phase6,
    select_live_subset_36,
    verify_frozen_manifest,
)
from scripts.run_phase6_evaluation import main as phase6_main

pytestmark = pytest.mark.adjudication

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_hash_preflight_detects_tampering(tmp_path: Path):
    """Modifying a frozen Phase 5 dataset file aborts verification immediately."""
    manifest_src = REPO_ROOT / "results" / "phase5" / "frozen_evaluation_manifest.json"
    info = verify_frozen_manifest(manifest_src, REPO_ROOT)
    assert len(info["verified_hashes"]) == 8

    # Copy dataset & manifest into tmp_path and mutate test_inputs.jsonl by 1 byte
    ds_dir = tmp_path / "datasets" / "frozen_v1"
    ds_dir.mkdir(parents=True)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    for f in (REPO_ROOT / "datasets" / "frozen_v1").glob("*.jsonl"):
        shutil.copy(f, ds_dir / f.name)
    for f in (REPO_ROOT / "docs").glob("FROZEN_*.md"):
        shutil.copy(f, docs_dir / f.name)

    manifest_copy = json.loads(manifest_src.read_text(encoding="utf-8"))
    for k, v in manifest_copy["hashes"].items():
        manifest_copy["hashes"][k]["path"] = v["path"].replace("\\", "/")
    m_path = tmp_path / "frozen_evaluation_manifest.json"
    m_path.write_text(json.dumps(manifest_copy), encoding="utf-8")

    # Tamper 1 byte in copied test_inputs.jsonl -> verify_frozen_manifest & CLI must abort!
    ti = ds_dir / "test_inputs.jsonl"
    ti.write_text(ti.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_frozen_manifest(m_path, tmp_path)

    rc = phase6_main([
        "--repo-root", str(tmp_path),
        "--manifest", str(m_path),
        "--results-dir", str(tmp_path / "out"),
        "--rebuild-tables-only",
    ])
    assert rc == 2


def test_live_subset_36_covers_all_8_families_and_blocked_when_key_unset(tmp_path: Path):
    """Stratified 36-claim subset covers all 8 test families and returns BLOCKED cleanly when key unset."""
    manifest_info = verify_frozen_manifest(
        REPO_ROOT / "results" / "phase5" / "frozen_evaluation_manifest.json", REPO_ROOT
    )
    test_inputs = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_inputs.jsonl")
    test_labels = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_oracle_labels.jsonl")
    upd_inputs = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_update_paths.jsonl")
    upd_labels = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_update_oracle_labels.jsonl")

    sub = select_live_subset_36(test_inputs, test_labels, upd_inputs)
    assert sub["claim_count"] == 36
    assert sub["update_count"] == 12
    assert len(sub["family_quotas"]) == 8
    assert set(sub["class_distribution"]) == {"APPROVE", "REJECT", "REVIEW"}

    blocked = run_live_phase6(
        subset_36=sub,
        test_labels_by_id={r["claim_id"]: r["label"] for r in test_labels},
        upd_labels_by_id={r["transition_id"]: r for r in upd_labels},
        raw_dir=tmp_path / "raw",
        provider="groq",
        model_id="openai/gpt-oss-20b",
        api_key_env_var="NONEXISTENT_PHASE6_KEY",
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["calls_used"] == 0
    assert not (tmp_path / "raw" / "live_claims.jsonl").exists()


def test_live_pipeline_contract_and_resume_skipping_with_labeled_offline_fake(tmp_path: Path):
    """EXPLICIT OFFLINE CONTRACT TEST ONLY: verifies Single-Agent, Judge, Recovery & --resume skipping."""
    manifest_info = verify_frozen_manifest(
        REPO_ROOT / "results" / "phase5" / "frozen_evaluation_manifest.json", REPO_ROOT
    )
    test_inputs = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_inputs.jsonl")
    test_labels = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_oracle_labels.jsonl")
    upd_inputs = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_update_paths.jsonl")
    upd_labels = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_update_oracle_labels.jsonl")

    full_sub = select_live_subset_36(test_inputs, test_labels, upd_inputs)
    # Trim subset to 3 claims + 2 updates for fast deterministic offline contract check
    mini_sub = {
        **full_sub,
        "claim_count": 3,
        "update_count": 2,
        "claims": full_sub["claims"][:3],
        "updates": [full_sub["updates"][0], full_sub["updates"][6]],
    }
    lbl_map = {r["claim_id"]: r["label"] for r in test_labels}
    upd_map = {r["transition_id"]: r for r in upd_labels}

    role_fake = FakeRoleSpecialistClient()
    call_counter = {"n": 0}

    def offline_fake_router(messages, profile):
        call_counter["n"] += 1
        sys_txt = messages[0]["content"]
        if "Claim Facts:" in messages[-1]["content"]:
            facts_d = json.loads(messages[-1]["content"].split("Claim Facts: ", 1)[1])
            c = InsuranceClaim(claim_id="tmp", **facts_d)
            rules = [rule_decision(r, c.facts(r)) for r in ("policy", "evidence", "risk", "compliance")]
            dec = "REJECT" if "REJECT" in rules else ("REVIEW" if "REVIEW" in rules else "APPROVE")
            return json.dumps({"decision": dec, "confidence": 0.9, "reasoning_summary": "[OFFLINE_FAKE_E2E]"})
        if "Specialist Outputs:" in messages[-1]["content"]:
            spec_d = json.loads(messages[-1]["content"].split("Specialist Outputs: ", 1)[1])
            decs = [v.get("decision") for v in spec_d.values()]
            dec = "REJECT" if spec_d["policy"].get("decision") == "REJECT" else ("REVIEW" if "REVIEW" in decs else "APPROVE")
            return json.dumps({"decision": dec, "confidence": 0.9, "reasoning_summary": "[OFFLINE_FAKE_JUDGE]"})
        return role_fake(messages, profile)

    raw_dir = tmp_path / "raw"
    res1 = run_live_phase6(
        subset_36=mini_sub,
        test_labels_by_id=lbl_map,
        upd_labels_by_id=upd_map,
        raw_dir=raw_dir,
        provider="groq",
        model_id="openai/gpt-oss-20b",
        api_key_env_var="GROQ_API_KEY",
        resume=True,
        client_override=offline_fake_router,
    )
    assert res1["status"] == "PASS"
    assert res1["completed_full_live_claims"] == 3
    assert res1["completed_single_agent_claims"] == 3
    assert res1["completed_judge_claims"] == 3
    assert res1["completed_recovery_paths"] == 2
    first_calls = call_counter["n"]
    assert first_calls > 0

    # Resume must make ZERO additional model calls when all items are already persisted
    res2 = run_live_phase6(
        subset_36=mini_sub,
        test_labels_by_id=lbl_map,
        upd_labels_by_id=upd_map,
        raw_dir=raw_dir,
        provider="groq",
        model_id="openai/gpt-oss-20b",
        api_key_env_var="GROQ_API_KEY",
        resume=True,
        client_override=offline_fake_router,
    )
    assert call_counter["n"] == first_calls
    assert res2["completed_full_live_claims"] == 3

    # Table rebuilder produces consistent JSON purely from saved logs
    t1 = rebuild_phase6_tables(results_dir=tmp_path, manifest_info=manifest_info, repo_root=REPO_ROOT)
    t2 = rebuild_phase6_tables(results_dir=tmp_path, manifest_info=manifest_info, repo_root=REPO_ROOT)
    assert t1["live_arms"]["FULL_LIVE"]["cases_completed"] == 3
    assert t1["live_arms"]["FULL_LIVE"]["metrics"] == t2["live_arms"]["FULL_LIVE"]["metrics"]
    assert len(_single_agent_prompt(InsuranceClaim())) == 2


def test_live_builder_path_end_to_end_with_mocked_urlopen_and_pacing(monkeypatch, tmp_path: Path):
    """EXPLICIT OFFLINE CONTRACT TEST: exercises actual HostedChatModelClient + build_phase4_role_backends with min_call_interval_sec=5.0."""
    import io
    import re
    from cert_recovery.adjudication.domain import RULES

    manifest_info = verify_frozen_manifest(
        REPO_ROOT / "results" / "phase5" / "frozen_evaluation_manifest.json", REPO_ROOT
    )
    test_inputs = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_inputs.jsonl")
    test_labels = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_oracle_labels.jsonl")
    upd_inputs = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_update_paths.jsonl")
    upd_labels = _load_jsonl(REPO_ROOT / "datasets" / "frozen_v1" / "test_update_oracle_labels.jsonl")

    full_sub = select_live_subset_36(test_inputs, test_labels, upd_inputs)
    mini_sub = {
        **full_sub,
        "claim_count": 1,
        "update_count": 1,
        "claims": full_sub["claims"][:1],
        "updates": [full_sub["updates"][0]],
    }
    lbl_map = {r["claim_id"]: r["label"] for r in test_labels}
    upd_map = {r["transition_id"]: r for r in upd_labels}

    sleeps: list[float] = []
    now = {"t": 1000.0}

    def fake_sleep(sec: float):
        sleeps.append(sec)
        now["t"] += sec

    monkeypatch.setattr("cert_recovery.adjudication.model_agent.time.sleep", fake_sleep)
    monkeypatch.setattr("cert_recovery.adjudication.model_agent.time.monotonic", lambda: now["t"])
    monkeypatch.setattr("cert_recovery.oracle.phase6_eval.time.sleep", fake_sleep)
    monkeypatch.setenv("GROQ_API_KEY", "dummy-offline-test-key")

    http_calls = {"n": 0}

    class _FakeResp:
        def __init__(self, body_dict: dict):
            self._buf = io.BytesIO(json.dumps(body_dict).encode("utf-8"))

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=30.0):
        http_calls["n"] += 1
        req_json = json.loads(req.data.decode("utf-8"))
        msgs = req_json["messages"]
        sys_txt = msgs[0]["content"]
        user_txt = msgs[-1]["content"]

        if "Claim Facts:" in user_txt:
            facts_d = json.loads(user_txt.split("Claim Facts: ", 1)[1])
            c = InsuranceClaim(claim_id="tmp", **facts_d)
            rs = [rule_decision(r, c.facts(r)) for r in ("policy", "evidence", "risk", "compliance")]
            dec = "REJECT" if "REJECT" in rs else ("REVIEW" if "REVIEW" in rs else "APPROVE")
            content = json.dumps({"decision": dec, "confidence": 0.9, "reasoning_summary": "[MOCK_HTTP_E2E]"})
        elif "Specialist Outputs:" in user_txt:
            spec_d = json.loads(user_txt.split("Specialist Outputs: ", 1)[1])
            decs = [v.get("decision") for v in spec_d.values()]
            dec = "REJECT" if spec_d["policy"].get("decision") == "REJECT" else ("REVIEW" if "REVIEW" in decs else "APPROVE")
            content = json.dumps({"decision": dec, "confidence": 0.9, "reasoning_summary": "[MOCK_HTTP_JUDGE]"})
        else:
            role_match = re.search(r"Role:\s*(\w+)", sys_txt)
            role = role_match.group(1) if role_match else "compliance"
            facts_match = re.search(r"Facts[^:]*:\s*(\{.*\})", user_txt)
            facts = json.loads(facts_match.group(1))
            dec = rule_decision(role, facts)
            content = json.dumps({
                "decision": dec,
                "confidence": 0.95,
                "evidence": facts,
                "rule_references": [RULES[role][0]],
                "reasoning_summary": f"[MOCK_HTTP_{role.upper()}]",
            })

        return _FakeResp({
            "id": f"chatcmpl-{http_calls['n']}",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    raw_dir = tmp_path / "raw_http"
    res = run_live_phase6(
        subset_36=mini_sub,
        test_labels_by_id=lbl_map,
        upd_labels_by_id=upd_map,
        raw_dir=raw_dir,
        provider="groq",
        model_id="openai/gpt-oss-20b",
        api_key_env_var="GROQ_API_KEY",
        min_call_interval_sec=5.0,
        max_calls=50,
        resume=False,
        client_override=None,  # Exercises actual HostedChatModelClient + build_phase4_role_backends path
    )
    assert res["status"] == "PASS", res
    assert res["completed_full_live_claims"] == 1
    assert res["completed_single_agent_claims"] == 1
    assert res["completed_judge_claims"] == 1
    assert res["completed_recovery_paths"] == 1
    # 4 (FULL_LIVE) + 1 (SINGLE_AGENT) + 1 (JUDGE) + 4 (v1 build) + 1 (v2 narrow selective) + 4 (v2 full rerun) = 15 calls
    assert http_calls["n"] == 15
    assert res["budget"]["calls_used"] == 15
    # Verify 5.0s pacing applied across the 14 inter-call intervals
    assert len(sleeps) == 14
    assert all(s == pytest.approx(5.0) for s in sleeps)
