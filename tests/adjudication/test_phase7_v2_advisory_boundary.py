"""Phase 7 tests proving V2/semantic advice is strictly non-authoritative and V2 audit handles both present/missing runs."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from cert_recovery.adjudication.domain import InsuranceClaim
from cert_recovery.adjudication.independent_verifier import verify_composition_independently
from cert_recovery.adjudication.ledger import Ed25519EventSigner
from cert_recovery.adjudication.provenance_verifier import verify_persisted_run
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication
from cert_recovery.models import ArtifactKind, CertificateStatus, Ref
from remote_model_client import RemoteModelClient
from scripts.audit_phase7_v2_appendix import run_v2_audit
from semantic_advice import SemanticAdviceAdapter

pytestmark = pytest.mark.adjudication

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_html_nli_response(label_text: str, score_pct: str) -> str:
    return f"<div>{label_text}</div><div>Confidence: <strong>{score_pct}</strong></div>"


def test_semantic_advice_has_zero_authority_over_adjudication_certificates_and_recovery(tmp_path: Path):
    """Enabling or attaching adversarial V2/semantic advice leaves decisions, certificates, provenance, and recovery unchanged."""
    claim_v1 = InsuranceClaim()
    claim_v2 = replace(claim_v1, documents_available=False)

    signer_plain = Ed25519EventSigner.generate()
    signer_adv = Ed25519EventSigner.generate()

    # Baseline workflow WITHOUT any semantic advice
    wf_plain = SpecialistCompositionAdjudication(
        claim_v1,
        signer=signer_plain,
        provenance_dir=tmp_path / "plain",
        run_id="phase7-plain",
    )
    res_plain_v1 = wf_plain.result()

    # Identical workflow WITH adversarial high-confidence semantic advice attached
    wf_adv = SpecialistCompositionAdjudication(
        claim_v1,
        signer=signer_adv,
        provenance_dir=tmp_path / "with_advice",
        run_id="phase7-advice",
    )
    adapter = SemanticAdviceAdapter(wf_adv.system)
    prepared = adapter.prepare(wf_adv.system.latest("facts:evidence").ref, wf_adv.system.latest("evidence_agent").ref)
    rmc = RemoteModelClient(prepared.profile)
    rmc.client = Mock()
    rmc.client.submit.return_value.result.return_value = _make_html_nli_response("Contradiction", "100.00%")
    resp = rmc.infer(prepared.to_case(), run_id=prepared.request_id)
    annotation = adapter.consume(prepared, resp)

    assert annotation["disposition"] == "ACCEPTED"
    assert annotation["advisory_only"] is True
    assert annotation["dependency_judgment"] is None

    res_adv_v1 = wf_adv.result()
    assert res_adv_v1["decision"] == res_plain_v1["decision"] == "APPROVE"
    assert res_adv_v1["gate"] == res_plain_v1["gate"]
    assert wf_adv.system.certificate_status(Ref("round-certificate", 1)) == CertificateStatus.VALID

    # Independent verifier & signed EventDAG remain completely unaffected
    r_ref = wf_adv.system.latest("round").ref
    indep = verify_composition_independently(
        wf_adv.system.artifact(r_ref), wf_adv.system.provenance(r_ref), wf_adv.ledger
    )
    assert indep.passed is True
    pvr = verify_persisted_run(
        tmp_path / "with_advice" / "events.jsonl",
        signer_adv.public_key,
        tmp_path / "with_advice" / "checkpoint.json",
    )
    assert pvr.valid is True and pvr.checkpoint_verified is True

    # Selective recovery after claim_v1 -> claim_v2 executes identically regardless of advice
    outcome_plain = wf_plain.update_claim(claim_v2)
    outcome_adv = wf_adv.update_claim(claim_v2)
    assert wf_plain.executions == wf_adv.executions
    assert [r.artifact_id for r in outcome_plain.restored] == [r.artifact_id for r in outcome_adv.restored]
    assert wf_plain.result()["decision"] == wf_adv.result()["decision"] == "REVIEW"

    # Prior advice is rejected as stale after fact change without rescuing or blocking recovery
    stale_check = adapter.consume(prepared, resp)
    assert stale_check["disposition"] == "REJECTED_STALE"

    # Conversely, 100% entailment advice cannot pass a failing specialist/certificate verifier
    wf_bad = SpecialistCompositionAdjudication(claim_v2)
    bad_adapter = SemanticAdviceAdapter(wf_bad.system)
    prep_bad = bad_adapter.prepare(wf_bad.system.latest("facts:evidence").ref, wf_bad.system.latest("evidence_agent").ref)
    rmc_pos = RemoteModelClient(prep_bad.profile)
    rmc_pos.client = Mock()
    rmc_pos.client.submit.return_value.result.return_value = _make_html_nli_response("Consistent", "100.00%")
    pos_note = bad_adapter.consume(prep_bad, rmc_pos.infer(prep_bad.to_case(), run_id=prep_bad.request_id))
    assert pos_note["nli_relation"] == "entailment"
    assert wf_bad.result()["decision"] == "REVIEW"


def test_v2_audit_handles_missing_private_runs_gracefully(tmp_path: Path):
    """If _private/runs is absent in a public checkout, audit returns UNVERIFIED / MISSING without fabricating counts."""
    res = run_v2_audit(repo_root=tmp_path / "empty_checkout", output_dir=tmp_path / "out_missing")
    assert res["overall_status"] == "UNVERIFIED / MISSING"
    assert res["confusion_matrices"]["final_original"]["status"] == "UNVERIFIED / MISSING"
    assert res["main_paper_claims_affected"] == []
    assert (tmp_path / "out_missing" / "v2_appendix_summary.json").exists()
    assert (tmp_path / "out_missing" / "v2_appendix_verified.md").exists()
