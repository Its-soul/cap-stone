#!/usr/bin/env python3
"""Offline stale-certificate replay experiment through the live commit path.

Executes six controlled scenarios against `SpecialistCompositionAdjudication`
and `CertificateSystem.commit_decision` / `CertificateSystem.add_artifact`,
recording exact artifact/certificate bindings before and after transitions,
expected vs. actual verdicts, refusal grounds, and post-attempt engine state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cert_recovery.adjudication.domain import ROLES, InsuranceClaim
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication
from cert_recovery.models import ArtifactKind


def _snapshot_bindings(wf: SpecialistCompositionAdjudication) -> dict:
    """Capture current artifact/certificate versions, hashes, and statuses."""
    cert_ref = wf.system.latest("round-certificate").ref
    cert = wf.system.certificate(cert_ref)
    final_node = wf.system.latest("final")
    round_node = wf.system.latest("round")
    facts_bindings = {}
    agent_bindings = {}
    for role in ROLES:
        f_node = wf.system.latest(f"facts:{role}")
        a_node = wf.system.latest(f"{role}_agent")
        facts_bindings[role] = {
            "ref": str(f_node.ref),
            "version": f_node.ref.version,
            "content_hash": f_node.content_hash,
            "valid": wf.system.is_valid(f_node.ref),
        }
        agent_bindings[role] = {
            "ref": str(a_node.ref),
            "version": a_node.ref.version,
            "content_hash": a_node.content_hash,
            "valid": wf.system.is_valid(a_node.ref),
        }
    committed_events = [e for e in wf.system.audit_export() if e["kind"] == "decision_committed"]
    return {
        "epoch": wf.system.epoch,
        "facts": facts_bindings,
        "agents": agent_bindings,
        "round": {
            "ref": str(round_node.ref),
            "decision": round_node.payload["decision"],
            "gate": round_node.payload["gate"],
            "valid": wf.system.is_valid(round_node.ref),
        },
        "certificate": {
            "ref": str(cert.ref),
            "subject": str(cert.subject),
            "status": wf.system.certificate_status(cert.ref).value,
            "dependency_hash": cert.dependency_hash,
            "valid": wf.system.is_valid(cert.ref),
        },
        "final": {
            "ref": str(final_node.ref),
            "decision": final_node.payload["decision"],
            "valid": wf.system.is_valid(final_node.ref),
        },
        "committed_event_count": len(committed_events),
        "last_committed_ref": (
            f"{committed_events[-1]['payload']['ref']['artifact_id']}@{committed_events[-1]['payload']['ref']['version']}"
            if committed_events else None
        ),
    }


def _try_commit(wf: SpecialistCompositionAdjudication, final_ref) -> dict:
    try:
        payload = wf.system.commit_decision(final_ref)
        return {
            "accepted": True,
            "decision": payload["decision"],
            "error_type": None,
            "refusal_reason": None,
        }
    except Exception as exc:
        return {
            "accepted": False,
            "decision": None,
            "error_type": type(exc).__name__,
            "refusal_reason": str(exc),
        }


def _try_authorize_with_certificate(wf: SpecialistCompositionAdjudication, artifact_id: str, cert_ref, decision_label: str) -> dict:
    try:
        ref = wf.system.add_artifact(
            artifact_id,
            ArtifactKind.DECISION,
            {"decision": decision_label, "mode": "replay-attempt"},
            (cert_ref,),
        )
        return {
            "accepted": True,
            "created_ref": str(ref),
            "error_type": None,
            "refusal_reason": None,
        }
    except Exception as exc:
        return {
            "accepted": False,
            "created_ref": None,
            "error_type": type(exc).__name__,
            "refusal_reason": str(exc),
        }


def run_stale_certificate_experiment() -> dict:
    """Run all six stale-certificate replay scenarios through the live commit API."""
    scenarios: list[dict] = []

    # 1. Clean positive control (v1)
    wf1 = SpecialistCompositionAdjudication(InsuranceClaim())
    before_1 = _snapshot_bindings(wf1)
    final_1 = wf1.system.latest("final").ref
    attempt_1 = _try_commit(wf1, final_1)
    after_1 = _snapshot_bindings(wf1)
    scenarios.append({
        "scenario_id": "1_clean_positive_control",
        "description": "Issue v1 composition certificate and commit via live commit_decision API.",
        "expected_verdict": "ACCEPTED",
        "actual_verdict": "ACCEPTED" if attempt_1["accepted"] else "REFUSED",
        "passed": attempt_1["accepted"] is True and after_1["committed_event_count"] == 1,
        "bindings_before": before_1,
        "bindings_after": after_1,
        "attempt": attempt_1,
    })

    # 2. Stale commit replay (label changes APPROVE -> REVIEW)
    wf2 = SpecialistCompositionAdjudication(InsuranceClaim())
    final_v1_2 = wf2.system.latest("final").ref
    cert_v1_2 = wf2.system.latest("round-certificate").ref
    _try_commit(wf2, final_v1_2)
    before_2 = _snapshot_bindings(wf2)
    src_evi_2 = wf2.system.latest("facts:evidence")
    wf2.system.change_dependency(
        "facts:evidence",
        {**src_evi_2.payload, "facts": {"documents_available": False, "invoice_matches": True}},
    )
    attempt_2 = _try_commit(wf2, final_v1_2)
    after_2 = _snapshot_bindings(wf2)
    scenarios.append({
        "scenario_id": "2_stale_commit_replay_label_changes",
        "description": "Advance evidence v1 -> v2 (APPROVE -> REVIEW) and replay old v1 decision commit.",
        "expected_verdict": "REFUSED",
        "actual_verdict": "ACCEPTED" if attempt_2["accepted"] else "REFUSED",
        "passed": (
            attempt_2["accepted"] is False
            and after_2["certificate"]["status"] == "INVALID"
            and after_2["committed_event_count"] == 1
            and not wf2.system.is_valid(cert_v1_2)
        ),
        "bindings_before": before_2,
        "bindings_after": after_2,
        "attempt": attempt_2,
    })

    # 3. Stale authorization replay (label changes APPROVE -> REVIEW)
    wf3 = SpecialistCompositionAdjudication(InsuranceClaim())
    cert_v1_3 = wf3.system.latest("round-certificate").ref
    before_3 = _snapshot_bindings(wf3)
    src_evi_3 = wf3.system.latest("facts:evidence")
    wf3.system.change_dependency(
        "facts:evidence",
        {**src_evi_3.payload, "facts": {"documents_available": False, "invoice_matches": True}},
    )
    attempt_3_invalid = _try_authorize_with_certificate(wf3, "forged_v2_decision_invalid", cert_v1_3, "APPROVE")
    wf3._recover()
    attempt_3_superseded = _try_authorize_with_certificate(wf3, "forged_v2_decision_superseded", cert_v1_3, "APPROVE")
    after_3 = _snapshot_bindings(wf3)
    scenarios.append({
        "scenario_id": "3_stale_authorization_replay_label_changes",
        "description": "Attempt to authorize v2 decision using stale v1 certificate (both INVALID and SUPERSEDED).",
        "expected_verdict": "REFUSED",
        "actual_verdict": "ACCEPTED" if (attempt_3_invalid["accepted"] or attempt_3_superseded["accepted"]) else "REFUSED",
        "passed": (
            not attempt_3_invalid["accepted"]
            and not attempt_3_superseded["accepted"]
            and after_3["committed_event_count"] == 0
            and not wf3.system.is_valid(cert_v1_3)
        ),
        "bindings_before": before_3,
        "bindings_after": after_3,
        "attempt": {
            "while_invalid": attempt_3_invalid,
            "while_superseded": attempt_3_superseded,
            "refusal_reason": attempt_3_invalid["refusal_reason"],
        },
    })

    # 4. Same-label stale replay (APPROVE -> APPROVE and REJECT -> REJECT)
    # 4a: APPROVE (50,000) -> APPROVE (60,000)
    wf4a = SpecialistCompositionAdjudication(InsuranceClaim(amount_inr=50_000, coverage_limit_inr=100_000))
    final_v1_4a = wf4a.system.latest("final").ref
    cert_v1_4a = wf4a.system.latest("round-certificate").ref
    before_4a = _snapshot_bindings(wf4a)
    src_pol_4a = wf4a.system.latest("facts:policy")
    wf4a.system.change_dependency(
        "facts:policy",
        {**src_pol_4a.payload, "facts": {**src_pol_4a.payload["facts"], "amount_inr": 60_000}},
    )
    attempt_4a_commit = _try_commit(wf4a, final_v1_4a)
    attempt_4a_auth = _try_authorize_with_certificate(wf4a, "same_label_replay_approve", cert_v1_4a, "APPROVE")
    after_4a = _snapshot_bindings(wf4a)
    scenarios.append({
        "scenario_id": "4a_same_label_stale_replay_approve",
        "description": "Advance policy amount 50k -> 60k (APPROVE -> APPROVE) and replay v1 commit/certificate.",
        "expected_verdict": "REFUSED",
        "actual_verdict": "ACCEPTED" if (attempt_4a_commit["accepted"] or attempt_4a_auth["accepted"]) else "REFUSED",
        "passed": (
            not attempt_4a_commit["accepted"]
            and not attempt_4a_auth["accepted"]
            and after_4a["certificate"]["status"] == "INVALID"
            and after_4a["committed_event_count"] == 0
        ),
        "bindings_before": before_4a,
        "bindings_after": after_4a,
        "attempt": {
            "commit_replay": attempt_4a_commit,
            "auth_replay": attempt_4a_auth,
            "refusal_reason": attempt_4a_commit["refusal_reason"],
        },
    })

    # 4b: REJECT (not covered, 50,000) -> REJECT (not covered, 60,000)
    wf4b = SpecialistCompositionAdjudication(
        replace(InsuranceClaim(), hospitalization_covered=False, amount_inr=50_000)
    )
    final_v1_4b = wf4b.system.latest("final").ref
    cert_v1_4b = wf4b.system.latest("round-certificate").ref
    before_4b = _snapshot_bindings(wf4b)
    src_pol_4b = wf4b.system.latest("facts:policy")
    wf4b.system.change_dependency(
        "facts:policy",
        {**src_pol_4b.payload, "facts": {**src_pol_4b.payload["facts"], "amount_inr": 60_000}},
    )
    attempt_4b_commit = _try_commit(wf4b, final_v1_4b)
    attempt_4b_auth = _try_authorize_with_certificate(wf4b, "same_label_replay_reject", cert_v1_4b, "REJECT")
    after_4b = _snapshot_bindings(wf4b)
    scenarios.append({
        "scenario_id": "4b_same_label_stale_replay_reject",
        "description": "Advance excluded claim amount 50k -> 60k (REJECT -> REJECT) and replay v1 commit/certificate.",
        "expected_verdict": "REFUSED",
        "actual_verdict": "ACCEPTED" if (attempt_4b_commit["accepted"] or attempt_4b_auth["accepted"]) else "REFUSED",
        "passed": (
            not attempt_4b_commit["accepted"]
            and not attempt_4b_auth["accepted"]
            and after_4b["certificate"]["status"] == "INVALID"
            and after_4b["committed_event_count"] == 0
        ),
        "bindings_before": before_4b,
        "bindings_after": after_4b,
        "attempt": {
            "commit_replay": attempt_4b_commit,
            "auth_replay": attempt_4b_auth,
            "refusal_reason": attempt_4b_commit["refusal_reason"],
        },
    })

    # 5. Unrelated edit control
    wf5 = SpecialistCompositionAdjudication(InsuranceClaim())
    wf5.system.add_dependency("unrelated_audit_memo", {"note": "v1"})
    before_5 = _snapshot_bindings(wf5)
    cert_v1_5 = wf5.system.latest("round-certificate").ref
    final_v1_5 = wf5.system.latest("final").ref
    impact_5 = wf5.system.change_dependency("unrelated_audit_memo", {"note": "v2-unrelated-edit"})
    attempt_5 = _try_commit(wf5, final_v1_5)
    after_5 = _snapshot_bindings(wf5)
    scenarios.append({
        "scenario_id": "5_unrelated_edit_control",
        "description": "Edit unrelated dependency outside certificate closure and commit unaffected v1 certificate.",
        "expected_verdict": "ACCEPTED",
        "actual_verdict": "ACCEPTED" if attempt_5["accepted"] else "REFUSED",
        "passed": (
            attempt_5["accepted"] is True
            and cert_v1_5 in impact_5.unaffected_certificates
            and after_5["certificate"]["status"] == "VALID"
            and after_5["committed_event_count"] == 1
        ),
        "bindings_before": before_5,
        "bindings_after": after_5,
        "attempt": {
            **attempt_5,
            "unaffected_certificates": [str(r) for r in impact_5.unaffected_certificates],
        },
    })

    # 6. Post-recovery positive control
    wf6 = SpecialistCompositionAdjudication(InsuranceClaim())
    cert_v1_6 = wf6.system.latest("round-certificate").ref
    before_6 = _snapshot_bindings(wf6)
    src_evi_6 = wf6.system.latest("facts:evidence")
    wf6.system.change_dependency(
        "facts:evidence",
        {**src_evi_6.payload, "facts": {"documents_available": False, "invoice_matches": True}},
    )
    recovery_outcome = wf6._recover()
    final_v2_6 = wf6.system.latest("final").ref
    attempt_6 = _try_commit(wf6, final_v2_6)
    after_6 = _snapshot_bindings(wf6)
    scenarios.append({
        "scenario_id": "6_post_recovery_positive_control",
        "description": "Run selective recovery to issue fresh v2-bound certificate and commit cleanly.",
        "expected_verdict": "ACCEPTED",
        "actual_verdict": "ACCEPTED" if attempt_6["accepted"] else "REFUSED",
        "passed": (
            attempt_6["accepted"] is True
            and attempt_6["decision"] == "REVIEW"
            and recovery_outcome.status == "COMPLETE"
            and wf6.system.certificate_status(cert_v1_6).value == "SUPERSEDED"
            and after_6["certificate"]["ref"] == "round-certificate@2"
            and after_6["certificate"]["status"] == "VALID"
            and after_6["agents"]["policy"]["version"] == 1
            and after_6["agents"]["evidence"]["version"] == 2
        ),
        "bindings_before": before_6,
        "bindings_after": after_6,
        "attempt": {
            **attempt_6,
            "recovery_status": recovery_outcome.status,
            "restored": [str(r) for r in recovery_outcome.restored],
        },
    })

    return {
        "experiment": "stale-certificate-replay-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": len(scenarios),
        "passed_scenarios": sum(1 for s in scenarios if s["passed"]),
        "all_passed": all(s["passed"] for s in scenarios),
        "scenarios": scenarios,
    }


def write_replay_outputs(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stale_certificate_replay_matrix.json"
    md_path = output_dir / "stale_certificate_replay_matrix.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Stale-Certificate Replay Experiment Matrix",
        "",
        f"- Generated UTC: `{report['generated_utc']}`",
        f"- Result: `{report['passed_scenarios']}/{report['total_scenarios']}` scenarios passed (`all_passed={report['all_passed']}`)",
        "",
        "| Scenario | Expected | Actual | Cert Before -> After | Status Before -> After | Refusal Ground / Outcome | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for sc in report["scenarios"]:
        cb = sc["bindings_before"]["certificate"]
        ca = sc["bindings_after"]["certificate"]
        att = sc["attempt"]
        reason = att.get("refusal_reason") or f"COMMITTED ({att.get('decision', 'OK')})"
        lines.append(
            f"| `{sc['scenario_id']}` | `{sc['expected_verdict']}` | `{sc['actual_verdict']}` "
            f"| `{cb['ref']}` -> `{ca['ref']}` | `{cb['status']}` -> `{ca['status']}` "
            f"| `{reason}` | `{'PASS' if sc['passed'] else 'FAIL'}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "phase2",
        help="Destination directory for replay matrix artifacts.",
    )
    args = parser.parse_args()
    report = run_stale_certificate_experiment()
    json_path, md_path = write_replay_outputs(report, args.output_dir.resolve())
    print(f"Stale-certificate replay: {report['passed_scenarios']}/{report['total_scenarios']} passed")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
