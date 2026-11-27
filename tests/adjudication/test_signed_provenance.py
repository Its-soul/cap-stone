"""Tests for Ed25519-signed persisted EventDAG provenance, verifier CLI, tamper detection, and causal lineage."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cert_recovery.adjudication.domain import InsuranceClaim
from cert_recovery.adjudication.ledger import Ed25519EventSigner, EventDAG, _fsync_write_atomic
from cert_recovery.adjudication.provenance_verifier import (
    SIGNED_CHECKPOINT_KEYS,
    SIGNED_EVENT_KEYS,
    load_events_jsonl,
    query_decisions_for_artifact,
    verify_persisted_run,
)
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication
from cert_recovery.models import ArtifactKind, Ref

pytestmark = pytest.mark.adjudication

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ed25519_signed_persisted_dag_and_atomic_checkpoint(tmp_path: Path):
    """Persisted EventDAG writes append-only JSONL and atomic signed checkpoint.json."""
    signer = Ed25519EventSigner.generate()
    pub_hex_path = signer.write_public_key_hex(tmp_path / "public_key.hex")
    prov_dir = tmp_path / "run_prov"

    wf = SpecialistCompositionAdjudication(
        InsuranceClaim(),
        signer=signer,
        provenance_dir=prov_dir,
        run_id="unit-signed-run",
    )

    events_path = prov_dir / "events.jsonl"
    checkpoint_path = prov_dir / "checkpoint.json"
    assert events_path.exists()
    assert checkpoint_path.exists()
    # Ensure atomic write leaves no leftover .tmp files
    assert not list(prov_dir.glob("*.tmp"))

    records = load_events_jsonl(events_path)
    assert len(records) == len(wf.ledger.export())
    assert all(set(r) == SIGNED_EVENT_KEYS for r in records)
    assert [r["sequence"] for r in records] == list(range(len(records)))

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert set(checkpoint) == SIGNED_CHECKPOINT_KEYS
    assert checkpoint["event_count"] == len(records)
    assert checkpoint["head_sequence"] == len(records) - 1
    assert checkpoint["head_event_hash"] == records[-1]["event_hash"]
    assert checkpoint["signer_key_id"] == signer.key_id
    assert "final@1" in checkpoint["artifact_version_to_event_id"]
    assert "facts:evidence@1" in checkpoint["artifact_version_to_event_id"]

    vr = verify_persisted_run(events_path, pub_hex_path, checkpoint_path)
    assert vr.valid is True, vr.reason
    assert vr.checkpoint_verified is True
    assert vr.run_id == "unit-signed-run"


def test_fresh_process_restart_and_cli_verification(tmp_path: Path):
    """Verify persisted log in a fresh subprocess and verify CLI exit codes."""
    signer = Ed25519EventSigner.generate()
    pub_path = signer.write_public_key_hex(tmp_path / "public_key.hex")
    prov_dir = tmp_path / "persisted"

    SpecialistCompositionAdjudication(
        InsuranceClaim(),
        signer=signer,
        provenance_dir=prov_dir,
        run_id="cli-verify-run",
    )

    cli = REPO_ROOT / "scripts" / "verify_provenance.py"
    proc_ok = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--events",
            str(prov_dir / "events.jsonl"),
            "--public-key",
            str(pub_path),
            "--checkpoint",
            str(prov_dir / "checkpoint.json"),
            "--query-artifact",
            "facts:evidence@1",
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_ok.returncode == 0, proc_ok.stderr
    parsed = json.loads(proc_ok.stdout)
    assert parsed["valid"] is True
    assert parsed["checkpoint_verified"] is True
    assert parsed["lineage_query"]["dependent_decisions"] == ["final@1"]

    # Tamper one byte on disk and verify CLI exits non-zero in fresh subprocess
    raw = (prov_dir / "events.jsonl").read_text(encoding="utf-8")
    tampered = raw.replace('"APPROVE"', '"REJECT"', 1)
    assert tampered != raw
    (prov_dir / "events.jsonl").write_text(tampered, encoding="utf-8")

    proc_fail = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--events",
            str(prov_dir / "events.jsonl"),
            "--public-key",
            str(pub_path),
            "--checkpoint",
            str(prov_dir / "checkpoint.json"),
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_fail.returncode == 1
    parsed_fail = json.loads(proc_fail.stdout)
    assert parsed_fail["valid"] is False
    assert parsed_fail["error_code"] == "payload_digest_mismatch"


def test_atomic_checkpoint_preserves_prior_file_on_write_failure(tmp_path: Path, monkeypatch):
    """Simulate write/replace interruption: prior checkpoint remains intact and uncorrupted."""
    target = tmp_path / "checkpoint.json"
    _fsync_write_atomic(target, '{"state": "v1"}')

    def boom(src, dst):
        raise OSError("simulated power loss during replace")

    monkeypatch.setattr("cert_recovery.adjudication.ledger.os.replace", boom)
    with pytest.raises(OSError, match="simulated power loss"):
        _fsync_write_atomic(target, '{"state": "corrupt-v2"}')

    assert json.loads(target.read_text(encoding="utf-8")) == {"state": "v1"}
    assert not list(tmp_path.glob("*.tmp"))


def test_reopen_persisted_dag_continues_monotonic_sequence(tmp_path: Path):
    """Opening an existing persisted log appends from previous sequence and head hash."""
    signer = Ed25519EventSigner.generate()
    dag1 = EventDAG.open_persisted(tmp_path / "reopen", run_id="reopen-run", signer=signer)
    h0 = dag1.append("input_received", {"x": 1}, (), agent_id="facts:evidence", artifact_ref="facts:evidence@1", artifact_version=1)

    dag2 = EventDAG.open_persisted(tmp_path / "reopen", run_id="reopen-run", signer=signer)
    h1 = dag2.append("final_decision", {"decision": "APPROVE"}, (h0,), agent_id="final", artifact_ref="final@1", artifact_version=1)

    vr = verify_persisted_run(tmp_path / "reopen" / "events.jsonl", signer.public_key, tmp_path / "reopen" / "checkpoint.json")
    assert vr.valid is True, vr.reason
    assert vr.event_count == 2
    assert vr.head_event_hash == h1


def test_causal_lineage_query_and_artifact_dag_crosscheck(tmp_path: Path):
    """Cross-check EventDAG causal query against Artifact Dependency DAG across recovery."""
    signer = Ed25519EventSigner.generate()
    wf = SpecialistCompositionAdjudication(
        InsuranceClaim(),
        signer=signer,
        provenance_dir=tmp_path / "lineage",
        run_id="lineage-check",
    )

    def _artifact_deps(system, target: Ref) -> set[str]:
        out: set[str] = set()
        latest = system.latest("final").ref
        for v in range(1, latest.version + 1):
            d_ref = Ref("final", v)
            if system.artifact(d_ref).kind == ArtifactKind.DECISION and target in {
                n.ref for n in system.provenance(d_ref)
            }:
                out.add(f"final@{v}")
        return out

    # Pre-update boundary
    assert query_decisions_for_artifact(wf.ledger.export(), "facts:evidence@1") == {"final@1"}
    assert _artifact_deps(wf.system, Ref("facts:evidence", 1)) == {"final@1"}

    # Advance facts:evidence@1 -> facts:evidence@2 (documents_available=False => REVIEW)
    wf.update_claim(replace(InsuranceClaim(), documents_available=False))
    events_after = load_events_jsonl(tmp_path / "lineage" / "events.jsonl")
    assert query_decisions_for_artifact(events_after, "facts:evidence@1", include_operational_edges=False) == {"final@1"}
    assert _artifact_deps(wf.system, Ref("facts:evidence", 1)) == {"final@1"}
    assert query_decisions_for_artifact(events_after, "facts:policy@1", include_operational_edges=False) == {"final@1", "final@2"}
    assert _artifact_deps(wf.system, Ref("facts:policy", 1)) == {"final@1", "final@2"}
    # Operational recovery edge traversal explains naive over-approximation
    assert query_decisions_for_artifact(events_after, "facts:evidence@1", include_operational_edges=True) == {
        "final@1",
        "final@2",
    }

    # Agent recovery where facts:evidence@1 IS genuinely reused
    wf_reuse = SpecialistCompositionAdjudication(
        InsuranceClaim(),
        signer=signer,
        provenance_dir=tmp_path / "lineage_reuse",
        run_id="lineage-reuse",
    )
    wf_reuse.recover_agent("risk_agent")
    assert query_decisions_for_artifact(wf_reuse.ledger.export(), "facts:evidence@1") == {"final@1", "final@2"}
    assert _artifact_deps(wf_reuse.system, Ref("facts:evidence", 1)) == {"final@1", "final@2"}


def test_tamper_experiment_runner_11_cases_all_pass(tmp_path: Path):
    """Run the 11-case tamper/replay/rollback suite and assert all expected detections and limit control."""
    from scripts.run_provenance_tamper_experiment import run_experiment

    report = run_experiment(tmp_path / "phase3_out")
    s = report["summary"]
    assert s["total_cases"] == 11
    assert s["positive_controls"] == 1
    assert s["tamper_attacks_attempted"] == 9
    assert s["tamper_attacks_detected"] == 9
    assert s["tamper_attacks_accepted"] == 0
    assert s["honest_limit_accepted_as_expected"] == 1
    assert s["invalid_test"] == 0
    assert s["all_cases_passed"] is True

    c11 = next(c for c in report["cases"] if c["case_id"] == "C11_checkpoint_rollback")
    assert c11["colocated_rollback_snapshot_valid"] is True
    assert c11["actual_valid"] is False
    assert c11["error_code"] == "checkpoint_event_count_mismatch"


def test_checkpoint_rollback_attack_and_cli_rejection(tmp_path: Path):
    """Rolling back both events.jsonl and checkpoint.json to v1 is rejected against retained v2 checkpoint."""
    signer = Ed25519EventSigner.generate()
    pub_path = signer.write_public_key_hex(tmp_path / "public_key.hex")
    prov_dir = tmp_path / "rollback_run"

    wf = SpecialistCompositionAdjudication(
        InsuranceClaim(),
        signer=signer,
        provenance_dir=prov_dir,
        run_id="rollback-unit-run",
    )
    events_v1_bytes = (prov_dir / "events.jsonl").read_bytes()
    checkpoint_v1_bytes = (prov_dir / "checkpoint.json").read_bytes()

    # Continue run to produce newer independently retained expected_checkpoint_v2.json
    wf.update_claim(replace(InsuranceClaim(), documents_available=False))
    expected_ckpt_v2 = tmp_path / "expected_checkpoint_v2.json"
    expected_ckpt_v2.write_bytes((prov_dir / "checkpoint.json").read_bytes())

    # Roll back on-disk run directory to genuine v1 (events.jsonl, checkpoint.json)
    (prov_dir / "events.jsonl").write_bytes(events_v1_bytes)
    (prov_dir / "checkpoint.json").write_bytes(checkpoint_v1_bytes)

    # Confirm rolled-back co-located v1 pair itself is validly signed before checking retained v2 anchor
    vr_v1 = verify_persisted_run(prov_dir / "events.jsonl", pub_path, prov_dir / "checkpoint.json")
    assert vr_v1.valid is True
    assert vr_v1.checkpoint_verified is True

    # Confirm direct verification and standalone CLI detect & reject rollback against retained v2 checkpoint
    vr_rollback = verify_persisted_run(prov_dir / "events.jsonl", pub_path, expected_ckpt_v2)
    assert vr_rollback.valid is False
    assert vr_rollback.error_code == "checkpoint_event_count_mismatch"

    cli = REPO_ROOT / "scripts" / "verify_provenance.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--events",
            str(prov_dir / "events.jsonl"),
            "--public-key",
            str(pub_path),
            "--checkpoint",
            str(expected_ckpt_v2),
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    out = json.loads(proc.stdout)
    assert out["valid"] is False
    assert out["error_code"] == "checkpoint_event_count_mismatch"
