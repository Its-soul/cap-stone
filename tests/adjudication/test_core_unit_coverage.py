"""Focused Phase 1 unit tests for validator, composer, invalidation, recovery, and oracle guard."""
from __future__ import annotations

from dataclasses import replace
import pytest

from cert_recovery import ArtifactKind as K, CertificateStatus as S, CertificateSystem, VerificationResult
from cert_recovery.adjudication.composition import (
    COMPOSITION_CONTRACT_VERSION,
    PRECEDENCE_STRICT_AUDIT,
    PRECEDENCE_SUBSTANTIVE_EXCLUSION,
    assess_specialist,
    compose_specialist_assessments,
    verify_specialist_composition,
)
from cert_recovery.adjudication.consensus import assess
from cert_recovery.adjudication.domain import InsuranceClaim, PROMPTS, ROLES, RuleAgent
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication

pytestmark = pytest.mark.adjudication


def _valid_raw(role: str, claim: InsuranceClaim, binding: dict) -> dict:
    return RuleAgent().execute(role, PROMPTS[role], claim.facts(role), binding)


# (a) Validator unit tests
def test_validator_rejects_bool_confidence_nan_and_role_mismatch():
    claim = InsuranceClaim()
    facts = claim.facts("policy")
    binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "h1"}
    raw = _valid_raw("policy", claim, binding)

    for bad_conf in (True, False, float("nan"), float("inf"), -0.01, 1.01, "0.9"):
        bad = {**raw, "confidence": bad_conf}
        assert assess(bad, "policy", facts, binding).validation_status == "FAILED"
        spec = assess_specialist(bad, "policy", facts, binding)
        assert spec.validation_status == "FAILED"
        assert spec.obligation_result == "UNCERTAIN"
        assert spec.rule_consequence == "REVIEW"

    # Wrong agent_id fails structural schema
    wrong_id = {**raw, "agent_id": "evidence_agent"}
    assert assess_specialist(wrong_id, "policy", facts, binding).validation_status == "FAILED"

    # Impossible consequence + high confidence flags both reasons
    bad_pol = {**raw, "decision": "REVIEW", "confidence": 0.99}
    spec_pol = assess_specialist(bad_pol, "policy", facts, binding)
    assert spec_pol.validation_status == "INVALID"
    assert "impossible_role_consequence:REVIEW_not_permitted_for_policy" in spec_pol.reasons
    assert "high_confidence_invalid_output" in spec_pol.reasons


# (b) Composer unit tests
def test_composer_truth_table_and_verifier_tamper_guards():
    claim = replace(InsuranceClaim(), hospitalization_covered=False)
    wf = SpecialistCompositionAdjudication(claim, faults={"evidence_agent": "failure"})
    round_node = wf.system.latest("round")
    agent_nodes = tuple(wf.system.artifact(ref) for ref in round_node.supports)

    # Valid composition round verifies cleanly
    vr = verify_specialist_composition(round_node.payload, agent_nodes, wf.ledger)
    assert vr.passed is True

    # Missing role in compose_specialist_assessments raises ValueError
    assessments = {}
    for role in ROLES:
        b = {"artifact_id": f"facts:{role}", "version": 1, "content_hash": "h"}
        assessments[role] = assess_specialist(_valid_raw(role, claim, b), role, claim.facts(role), b)
    with pytest.raises(ValueError, match="four roles"):
        compose_specialist_assessments({k: v for k, v in assessments.items() if k != "risk"})
    with pytest.raises(ValueError, match="Unsupported"):
        compose_specialist_assessments(assessments, precedence="unknown_mode")

    # Tampering composed decision fails verifier
    tampered = {**round_node.payload, "decision": "APPROVE"}
    assert not verify_specialist_composition(tampered, agent_nodes, wf.ledger).passed

    # Tampering contract version fails verifier
    tampered_ver = {**round_node.payload, "contract_version": "composition-v0"}
    assert not verify_specialist_composition(tampered_ver, agent_nodes, wf.ledger).passed


# (c) Invalidation unit tests
def test_invalidation_closure_and_unaffected_branch_isolation():
    system = CertificateSystem()
    system.register_verifier("ok", lambda subj, anc: VerificationResult(True, "ok"))
    a1 = system.add_dependency("branch_a_src", {"v": 1})
    b1 = system.add_dependency("branch_b_src", {"v": 10})
    a_claim = system.add_artifact("branch_a_claim", K.CLAIM, {"v": 1}, (a1,))
    b_claim = system.add_artifact("branch_b_claim", K.CLAIM, {"v": 10}, (b1,))
    cert_a = system.issue_certificate("cert_a", a_claim, "ok")
    cert_b = system.issue_certificate("cert_b", b_claim, "ok")

    report = system.change_dependency("branch_a_src", {"v": 2})
    assert cert_a in report.transitive_certificates
    assert cert_b in report.unaffected_certificates
    assert not system.is_valid(a_claim)
    assert not system.is_valid(cert_a)
    assert system.is_valid(b_claim)
    assert system.is_valid(cert_b)
    assert system.certificate_status(cert_a) == S.INVALID
    assert system.certificate_status(cert_b) == S.VALID


# (d) Recovery unit tests
def test_recovery_preserves_unaffected_certificates_and_blocks_missing_rebuilder():
    claim = InsuranceClaim()
    wf = SpecialistCompositionAdjudication(claim)
    old_policy = wf.system.latest("policy_agent")
    old_risk = wf.system.latest("risk_agent")
    old_comp = wf.system.latest("compliance_agent")

    # Single-branch recovery preserves the other 3 specialists at v1
    outcome = wf.update_claim(replace(claim, documents_available=False))
    assert outcome.status == "COMPLETE"
    assert wf.system.latest("evidence_agent").ref.version == 2
    assert wf.system.latest("policy_agent") == old_policy
    assert wf.system.latest("risk_agent") == old_risk
    assert wf.system.latest("compliance_agent") == old_comp
    assert wf.result()["decision"] == "REVIEW"

    # Deleting a rebuilder blocks recovery and prevents committing stale decision
    del wf.system._recipes["risk_agent"]
    report = wf.system.change_dependency(
        "facts:risk",
        {**wf.system.latest("facts:risk").payload, "facts": {"duplicate_claim": True, "provider_flagged": False}},
    )
    blocked_outcome = wf.system.execute_recovery(wf.system.plan_recovery(report))
    assert blocked_outcome.status == "BLOCKED"
    with pytest.raises(ValueError, match="stale/invalid"):
        wf.result()
