#!/usr/bin/env python3
"""Offline tamper, replay, rollback, and causal lineage experiment for EventDAG.

Executes controlled fault-injection and cross-check experiments against real
`SpecialistCompositionAdjudication` runs persisted with Ed25519 signatures,
append-only `events.jsonl`, and atomically replaced `checkpoint.json`.

No private signing keys are persisted to disk or printed in outputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cert_recovery.adjudication.domain import InsuranceClaim  # noqa: E402
from cert_recovery.adjudication.ledger import Ed25519EventSigner  # noqa: E402
from cert_recovery.adjudication.provenance_verifier import (  # noqa: E402
    TRUST_MODEL_SUMMARY,
    canonical_event_body,
    canonical_event_signing_bytes,
    load_events_jsonl,
    query_decisions_for_artifact,
    verify_persisted_run,
)
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication  # noqa: E402
from cert_recovery.crypto import sha256  # noqa: E402
from cert_recovery.models import ArtifactKind, Ref  # noqa: E402


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _artifact_dag_decisions(wf: SpecialistCompositionAdjudication, target_ref: Ref) -> set[str]:
    """Query Artifact Dependency DAG using public `system.provenance` across DECISION nodes."""
    hits: set[str] = set()
    latest_final = wf.system.latest("final").ref
    for version in range(1, latest_final.version + 1):
        dec_ref = Ref("final", version)
        node = wf.system.artifact(dec_ref)
        if node.kind != ArtifactKind.DECISION:
            continue
        ancestors = {a.ref for a in wf.system.provenance(dec_ref)}
        if target_ref in ancestors:
            hits.add(f"{dec_ref.artifact_id}@{dec_ref.version}")
    return hits


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    persisted_dir = output_dir / "persisted_run"
    reuse_dir = output_dir / "reuse_run"
    foreign_dir = output_dir / "foreign_run"
    cases_dir = output_dir / "tamper_cases"
    for d in (persisted_dir, reuse_dir, foreign_dir, cases_dir):
        if d.exists():
            shutil.rmtree(d)
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Ephemeral Ed25519 key in memory only; never write private key material to disk.
    signer = Ed25519EventSigner.generate()
    pub_key_path = signer.write_public_key_hex(persisted_dir / "public_key.hex")

    claim_v1 = InsuranceClaim()
    wf = SpecialistCompositionAdjudication(
        claim_v1,
        signer=signer,
        provenance_dir=persisted_dir,
        run_id="tamper-exp-v1",
    )

    events_path = persisted_dir / "events.jsonl"
    checkpoint_path = persisted_dir / "checkpoint.json"
    pre_update_events = load_events_jsonl(events_path)
    pre_update_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    # Pre-update causal lineage cross-check (before v1 -> v2 mutation)
    pre_event_dag_deps = sorted(
        query_decisions_for_artifact(pre_update_events, "facts:evidence@1")
    )
    pre_artifact_dag_deps = sorted(_artifact_dag_decisions(wf, Ref("facts:evidence", 1)))

    # Advance facts:evidence v1 -> v2 (documents_available=False => REVIEW at v2)
    claim_v2 = replace(claim_v1, documents_available=False)
    wf.update_claim(claim_v2)

    full_events = load_events_jsonl(events_path)
    full_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    # Post-update causal lineage cross-check (evidence changed v1 -> v2)
    post_event_dag_data_deps = sorted(
        query_decisions_for_artifact(full_events, "facts:evidence@1", include_operational_edges=False)
    )
    post_event_dag_operational_deps = sorted(
        query_decisions_for_artifact(full_events, "facts:evidence@1", include_operational_edges=True)
    )
    post_artifact_dag_deps = sorted(_artifact_dag_decisions(wf, Ref("facts:evidence", 1)))

    # Reuse control run: recover risk_agent without changing facts:evidence@1
    wf_reuse = SpecialistCompositionAdjudication(
        claim_v1,
        signer=signer,
        provenance_dir=reuse_dir,
        run_id="reuse-check-v1",
    )
    wf_reuse.recover_agent("risk_agent")
    reuse_events = load_events_jsonl(reuse_dir / "events.jsonl")
    reuse_event_dag_deps = sorted(
        query_decisions_for_artifact(reuse_events, "facts:evidence@1", include_operational_edges=False)
    )
    reuse_artifact_dag_deps = sorted(_artifact_dag_decisions(wf_reuse, Ref("facts:evidence", 1)))

    # Foreign run signed with the SAME key for cross-run splice test
    _ = SpecialistCompositionAdjudication(
        claim_v1,
        signer=signer,
        provenance_dir=foreign_dir,
        run_id="tamper-exp-foreign",
    )
    foreign_events = load_events_jsonl(foreign_dir / "events.jsonl")

    # Helper to construct, persist, and verify each case cleanly
    case_specs: list[dict[str, Any]] = []

    def _run_case(
        case_id: str,
        kind: str,
        title: str,
        records: list[dict[str, Any]],
        checkpoint_obj: dict[str, Any] | None,
        expected_valid: bool,
    ) -> None:
        c_dir = cases_dir / case_id
        c_dir.mkdir(parents=True, exist_ok=True)
        e_path = c_dir / "events.jsonl"
        _write_jsonl(e_path, records)
        cp_path: Path | None = None
        if checkpoint_obj is not None:
            cp_path = c_dir / "checkpoint.json"
            cp_path.write_text(json.dumps(checkpoint_obj, indent=2, sort_keys=True), encoding="utf-8")

        res = verify_persisted_run(e_path, pub_key_path, cp_path)
        detected = int(not res.valid)
        accepted = int(res.valid)
        passed = res.valid == expected_valid
        case_specs.append({
            "case_id": case_id,
            "kind": kind,
            "title": title,
            "expected_valid": expected_valid,
            "actual_valid": res.valid,
            "detected": detected,
            "accepted": accepted,
            "invalid_test": 0,
            "error_code": res.error_code,
            "reason": res.reason,
            "checkpoint_provided": checkpoint_obj is not None,
            "checkpoint_verified": res.checkpoint_verified,
            "status": "PASS" if passed else "FAIL",
        })

    # Case 1: Clean positive control
    _run_case(
        "C01_clean_reload",
        "positive_control",
        "Clean persisted events.jsonl + retained checkpoint.json",
        deepcopy(full_events),
        deepcopy(full_checkpoint),
        expected_valid=True,
    )

    # Case 2: Raw payload byte tamper on final_decision
    c2 = deepcopy(full_events)
    final_idx = next(i for i, r in enumerate(c2) if r["event_type"] == "final_decision")
    c2[final_idx]["payload"]["decision"] = "TAMPERED_DECISION"
    _run_case(
        "C02_payload_byte_tamper",
        "tamper_attack",
        "Mutate final_decision payload without recomputing digests/signature",
        c2,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 3: Payload tamper + forged SHA-256 payload_digest & event_hash (old Ed25519 sig)
    c3 = deepcopy(full_events)
    c3[final_idx]["payload"]["decision"] = "FORGED_DECISION"
    c3[final_idx]["payload_digest"] = sha256(c3[final_idx]["payload"])
    c3[final_idx]["event_hash"] = sha256(canonical_event_body(c3[final_idx]))
    _run_case(
        "C03_forged_sha256_old_sig",
        "tamper_attack",
        "Recompute SHA-256 payload_digest and event_hash with original Ed25519 signature",
        c3,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 4: Wrong Ed25519 private key forgery
    c4 = deepcopy(full_events)
    rogue_signer = Ed25519EventSigner.generate()
    c4[final_idx]["payload"]["decision"] = "ROGUE_SIGNED"
    c4[final_idx]["payload_digest"] = sha256(c4[final_idx]["payload"])
    c4[final_idx]["signer_key_id"] = rogue_signer.key_id
    c4[final_idx]["event_hash"] = sha256(canonical_event_body(c4[final_idx]))
    c4[final_idx]["signature"] = rogue_signer.sign_bytes(canonical_event_signing_bytes(c4[final_idx]))
    _run_case(
        "C04_wrong_key_forgery",
        "tamper_attack",
        "Re-sign tampered event with rogue Ed25519 private key",
        c4,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 5: Causal parent_hashes DAG tamper
    c5 = deepcopy(full_events)
    join_idx = next(i for i, r in enumerate(c5) if len(r["parent_hashes"]) > 1)
    c5[join_idx]["parent_hashes"] = ["f" * 64]
    _run_case(
        "C05_parent_link_tamper",
        "tamper_attack",
        "Alter parent_hashes on interior multi-parent event",
        c5,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 6: Interior event deletion (sequence gap)
    c6 = deepcopy(full_events)
    del c6[3]
    _run_case(
        "C06_interior_deletion",
        "tamper_attack",
        "Delete interior event creating sequence gap",
        c6,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 7: Event reordering / swap
    c7 = deepcopy(full_events)
    c7[4], c7[5] = c7[5], c7[4]
    _run_case(
        "C07_event_reorder",
        "tamper_attack",
        "Swap two adjacent interior events",
        c7,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 8: Cross-run splice injection under SAME signer key
    c8 = deepcopy(full_events)
    spliced = deepcopy(foreign_events[2])
    c8[2] = spliced
    _run_case(
        "C08_cross_run_splice",
        "tamper_attack",
        "Splice validly signed event from foreign run_id under same signer key",
        c8,
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 9: Tail truncation / rollback WITH retained final checkpoint
    _run_case(
        "C09_truncation_with_checkpoint",
        "tamper_attack",
        "Truncate log back to pre-recovery prefix against retained final checkpoint.json",
        deepcopy(pre_update_events),
        deepcopy(full_checkpoint),
        expected_valid=False,
    )

    # Case 10: Honest trust-model control — valid-prefix tail truncation WITHOUT retained checkpoint
    _run_case(
        "C10_truncation_without_checkpoint_limit",
        "honest_limit_control",
        "Valid signed prefix verified WITHOUT retained checkpoint.json (accepted by design)",
        deepcopy(pre_update_events),
        None,
        expected_valid=True,
    )

    # Case 11: Checkpoint rollback — roll back BOTH events.jsonl and co-located checkpoint.json to v1,
    # while verifier checks against independently retained expected_checkpoint_v2.json.
    c11_dir = cases_dir / "C11_checkpoint_rollback"
    c11_dir.mkdir(parents=True, exist_ok=True)
    c11_events_path = c11_dir / "events.jsonl"
    c11_colocated_ckpt_path = c11_dir / "checkpoint.json"
    c11_retained_ckpt_path = c11_dir / "expected_checkpoint_v2.json"
    _write_jsonl(c11_events_path, deepcopy(pre_update_events))
    c11_colocated_ckpt_path.write_text(
        json.dumps(pre_update_checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    c11_retained_ckpt_path.write_text(
        json.dumps(full_checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    c11_colocated_res = verify_persisted_run(c11_events_path, pub_key_path, c11_colocated_ckpt_path)
    c11_retained_res = verify_persisted_run(c11_events_path, pub_key_path, c11_retained_ckpt_path)
    c11_valid_setup = c11_colocated_res.valid and c11_colocated_res.checkpoint_verified
    case_specs.append({
        "case_id": "C11_checkpoint_rollback",
        "kind": "tamper_attack",
        "title": "Roll back both events.jsonl and co-located checkpoint.json to v1 against retained v2 checkpoint",
        "expected_valid": False,
        "actual_valid": c11_retained_res.valid,
        "detected": int(not c11_retained_res.valid),
        "accepted": int(c11_retained_res.valid),
        "invalid_test": 0 if c11_valid_setup else 1,
        "error_code": c11_retained_res.error_code,
        "reason": f"{c11_retained_res.reason} (rolled-back co-located v1 pair validly signed={c11_valid_setup})",
        "checkpoint_provided": True,
        "checkpoint_verified": c11_retained_res.checkpoint_verified,
        "colocated_rollback_snapshot_valid": c11_valid_setup,
        "status": "PASS" if (not c11_retained_res.valid and c11_valid_setup) else "FAIL",
    })

    attacks = [c for c in case_specs if c["kind"] == "tamper_attack"]
    summary = {
        "total_cases": len(case_specs),
        "positive_controls": sum(1 for c in case_specs if c["kind"] == "positive_control"),
        "tamper_attacks_attempted": len(attacks),
        "tamper_attacks_detected": sum(c["detected"] for c in attacks),
        "tamper_attacks_accepted": sum(c["accepted"] for c in attacks),
        "honest_limit_controls": sum(1 for c in case_specs if c["kind"] == "honest_limit_control"),
        "honest_limit_accepted_as_expected": sum(
            c["accepted"] for c in case_specs if c["kind"] == "honest_limit_control" and c["status"] == "PASS"
        ),
        "invalid_test": sum(c["invalid_test"] for c in case_specs),
        "all_cases_passed": all(c["status"] == "PASS" for c in case_specs),
    }

    causal_crosscheck = {
        "query": "Which decisions depended on facts:evidence@1?",
        "pre_update_boundary": {
            "event_dag_decisions": pre_event_dag_deps,
            "artifact_dag_decisions": pre_artifact_dag_deps,
            "exact_match": pre_event_dag_deps == pre_artifact_dag_deps,
        },
        "post_evidence_update_v1_to_v2": {
            "artifact_dag_decisions": post_artifact_dag_deps,
            "event_dag_data_lineage_decisions": post_event_dag_data_deps,
            "event_dag_including_operational_recovery_edges": post_event_dag_operational_deps,
            "data_lineage_matches_artifact_dag": post_event_dag_data_deps == post_artifact_dag_deps,
            "discrepancy_diagnosis": (
                "recovery_started links old final_decision (final@1) into recovery execution "
                "for audit ordering, so traversing operational edges reaches final@2 even though "
                "final@2 uses facts:evidence@2; stopping data-dependency traversal at "
                "recovery_started/recovery yields exact parity (['final@1'])."
            ),
        },
        "post_risk_agent_recovery_reusing_evidence_v1": {
            "artifact_dag_decisions": reuse_artifact_dag_deps,
            "event_dag_data_lineage_decisions": reuse_event_dag_deps,
            "exact_match": reuse_event_dag_deps == reuse_artifact_dag_deps,
        },
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "tamper-exp-v1",
        "signer_key_id": signer.key_id,
        "persisted_events_count": len(full_events),
        "pre_update_events_count": len(pre_update_events),
        "persisted_checkpoint": {
            "head_sequence": full_checkpoint["head_sequence"],
            "event_count": full_checkpoint["event_count"],
            "head_event_hash": full_checkpoint["head_event_hash"],
        },
        "pre_update_checkpoint": {
            "head_sequence": pre_update_checkpoint["head_sequence"],
            "event_count": pre_update_checkpoint["event_count"],
            "head_event_hash": pre_update_checkpoint["head_event_hash"],
        },
        "summary": summary,
        "cases": case_specs,
        "trust_model": TRUST_MODEL_SUMMARY,
    }

    matrix_json = output_dir / "provenance_tamper_matrix.json"
    matrix_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    causal_json = output_dir / "causal_crosscheck.json"
    causal_json.write_text(json.dumps(causal_crosscheck, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        "# Phase 3 Provenance Tamper, Replay & Rollback Matrix",
        "",
        f"- **Generated (UTC):** `{report['generated_utc']}`",
        f"- **Run ID:** `{report['run_id']}` (`{len(full_events)}` persisted events, `head_sequence={full_checkpoint['head_sequence']}`)",
        f"- **Signer Key ID (ephemeral public key only):** `{signer.key_id}`",
        "",
        "## Summary Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total Cases | {summary['total_cases']} |",
        f"| Positive Controls | {summary['positive_controls']} |",
        f"| Tamper / Replay / Rollback Attacks Attempted | {summary['tamper_attacks_attempted']} |",
        f"| Tamper Attacks Detected | {summary['tamper_attacks_detected']} |",
        f"| Tamper Attacks Accepted | {summary['tamper_attacks_accepted']} |",
        f"| Honest Limit Controls (No Checkpoint) Accepted | {summary['honest_limit_accepted_as_expected']} |",
        f"| Invalid Tests | {summary['invalid_test']} |",
        f"| All Cases Expected-Behavior Match | `{summary['all_cases_passed']}` |",
        "",
        "## Case Matrix",
        "",
        "| Case ID | Kind | Checkpoint | Expected Valid | Actual Valid | Error Code | Detected | Accepted | Status |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for c in case_specs:
        md_lines.append(
            f"| `{c['case_id']}` | `{c['kind']}` | `{c['checkpoint_provided']}` | `{c['expected_valid']}` | `{c['actual_valid']}` | `{c['error_code'] or 'none'}` | {c['detected']} | {c['accepted']} | **{c['status']}** |"
        )
    md_lines.extend([
        "",
        "## Causal Lineage Cross-Check (`facts:evidence@1`)",
        "",
        f"- **Pre-update boundary:** Artifact DAG `{pre_artifact_dag_deps}` vs EventDAG `{pre_event_dag_deps}` (match=`{causal_crosscheck['pre_update_boundary']['exact_match']}`)",
        f"- **Post `facts:evidence@1 -> @2` recovery:** Artifact DAG `{post_artifact_dag_deps}` vs EventDAG (data lineage) `{post_event_dag_data_deps}` (match=`{causal_crosscheck['post_evidence_update_v1_to_v2']['data_lineage_matches_artifact_dag']}`); EventDAG with operational `recovery_started` edges `{post_event_dag_operational_deps}`",
        f"- **Post `risk_agent` recovery reusing `facts:evidence@1`:** Artifact DAG `{reuse_artifact_dag_deps}` vs EventDAG `{reuse_event_dag_deps}` (match=`{causal_crosscheck['post_risk_agent_recovery_reusing_evidence_v1']['exact_match']}`)",
        "",
    ])
    (output_dir / "provenance_tamper_matrix.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 provenance tamper and causal lineage experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3"))
    args = parser.parse_args()
    report = run_experiment(args.output_dir)
    if not report["summary"]["all_cases_passed"]:
        print("FAILED: Not all provenance tamper cases matched expected behavior", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
