"""Comprehensive test suite for specialist-role composition prototype (composition-v1)."""
from dataclasses import asdict, replace
import pytest

from cert_recovery.adjudication.composition import (
    COMPOSITION_CONTRACT_VERSION,
    IMPOSSIBLE_CONSEQUENCES,
    OBLIGATION_NAMES,
    PERMITTED_CONSEQUENCES,
    PRECEDENCE_STRICT_AUDIT,
    PRECEDENCE_SUBSTANTIVE_EXCLUSION,
    ROLE_CONTRACTS,
    SpecialistAssessment,
    assess_specialist,
    compose_specialist_assessments,
)
from cert_recovery.adjudication.domain import InsuranceClaim, ROLES, RULES
from cert_recovery.adjudication.workflow import InsuranceAdjudication, SpecialistCompositionAdjudication


def test_clean_approval_all_obligations_satisfied():
    claim = InsuranceClaim()
    workflow = SpecialistCompositionAdjudication(claim)
    result = workflow.result()

    assert result["decision"] == "APPROVE"
    assert result["gate"] == "ALL_OBLIGATIONS_SATISFIED"
    assert result["mode"] == "specialist-composition"
    assert result["contract_version"] == COMPOSITION_CONTRACT_VERSION
    assert result["precedence"] == PRECEDENCE_SUBSTANTIVE_EXCLUSION

    # Verify certificate integrity
    from cert_recovery.models import CertificateStatus
    cert_ref = workflow.system.latest("round-certificate").ref
    assert workflow.system.certificate_status(cert_ref) == CertificateStatus.VALID

    # Verify each specialist was validated and satisfied
    round_payload = workflow.system.latest("round").payload
    assessments = round_payload["specialist_assessments"]
    for role in ROLES:
        assert assessments[role]["obligation_result"] == "SATISFIED"
        assert assessments[role]["rule_consequence"] == "APPROVE"
        assert assessments[role]["validation_status"] == "VALID"
        assert assessments[role]["validation_score"] == 1.0
        assert assessments[role]["fault_score"] == 0.0


@pytest.mark.parametrize("patch,field_name", [
    ({"hospitalization_covered": False}, "hospitalization_covered"),
    ({"waiting_period_satisfied": False}, "waiting_period_satisfied"),
    ({"amount_inr": 150_000, "coverage_limit_inr": 100_000}, "coverage_limit"),
])
def test_each_policy_exclusion_yields_reject(patch, field_name):
    claim = replace(InsuranceClaim(), **patch)
    workflow = SpecialistCompositionAdjudication(claim)
    result = workflow.result()

    # Confirmed policy exclusion is authoritative REJECT
    assert result["decision"] == "REJECT", f"Failed for {field_name}"
    assert result["gate"] == "SUBSTANTIVE_EXCLUSION"

    round_payload = workflow.system.latest("round").payload
    assessments = round_payload["specialist_assessments"]
    assert assessments["policy"]["obligation_result"] == "FAILED"
    assert assessments["policy"]["rule_consequence"] == "REJECT"
    assert assessments["policy"]["validation_status"] == "VALID"

    # Non-policy specialists remain satisfied (no role-fusion override!)
    for role in ("evidence", "risk", "compliance"):
        assert assessments[role]["obligation_result"] == "SATISFIED"
        assert assessments[role]["rule_consequence"] == "APPROVE"


@pytest.mark.parametrize("patch,trigger_role", [
    ({"documents_available": False}, "evidence"),
    ({"invoice_matches": False}, "evidence"),
    ({"duplicate_claim": True}, "risk"),
    ({"provider_flagged": True}, "risk"),
    ({"identity_verified": False}, "compliance"),
    ({"consent_available": False}, "compliance"),
])
def test_each_procedural_review_trigger(patch, trigger_role):
    claim = replace(InsuranceClaim(), **patch)
    workflow = SpecialistCompositionAdjudication(claim)
    result = workflow.result()

    assert result["decision"] == "REVIEW"
    assert result["gate"] == "PROCEDURAL_DEFERRAL"

    round_payload = workflow.system.latest("round").payload
    assessments = round_payload["specialist_assessments"]
    assert assessments["policy"]["obligation_result"] == "SATISFIED"
    assert assessments[trigger_role]["obligation_result"] == "FAILED"
    assert assessments[trigger_role]["rule_consequence"] == "REVIEW"


def test_mixed_exclusion_and_review_conditions():
    # Substantive exclusion (waiting period unsatisfied) + missing documents
    claim = replace(InsuranceClaim(), waiting_period_satisfied=False, documents_available=False)

    # Under substantive exclusion precedence: REJECT takes precedence over procedural missing docs
    wf_substantive = SpecialistCompositionAdjudication(claim, precedence="substantive_exclusion")
    res_substantive = wf_substantive.result()
    assert res_substantive["decision"] == "REJECT"
    assert res_substantive["gate"] == "SUBSTANTIVE_EXCLUSION"
    assert "evidence:FAILED(REVIEW)" in res_substantive["procedural_exceptions"]

    # Under strict audit precedence: neither is UNCERTAIN, policy is FAILED -> REJECT
    wf_strict = SpecialistCompositionAdjudication(claim, precedence="strict_audit")
    res_strict = wf_strict.result()
    assert res_strict["decision"] == "REJECT"
    assert res_strict["gate"] == "SUBSTANTIVE_EXCLUSION"


def test_mixed_exclusion_with_execution_uncertainty():
    # Substantive exclusion with crashed evidence backend (uncertainty)
    claim = replace(InsuranceClaim(), hospitalization_covered=False)

    # Substantive exclusion: exclusion is authoritative; crashed non-policy backend logged
    wf_sub = SpecialistCompositionAdjudication(claim, faults={"evidence_agent": "failure"}, precedence="substantive_exclusion")
    res_sub = wf_sub.result()
    assert res_sub["decision"] == "REJECT"
    assert res_sub["gate"] == "SUBSTANTIVE_EXCLUSION"
    assert any("evidence:UNCERTAIN(REVIEW)" in e for e in res_sub["procedural_exceptions"])

    # Strict audit: any UNCERTAIN halts decision into REVIEW
    wf_strict = SpecialistCompositionAdjudication(claim, faults={"evidence_agent": "failure"}, precedence="strict_audit")
    res_strict = wf_strict.result()
    assert res_strict["decision"] == "REVIEW"
    assert res_strict["gate"] == "AUDIT_UNCERTAINTY"


def test_impossible_consequences_rejected():
    base_facts = InsuranceClaim().facts("risk")
    binding = {"artifact_id": "facts:risk", "version": 1, "content_hash": "dummy"}

    # Risk asserting REJECT (screening flag cannot prove fraud)
    raw_risk_reject = {
        "agent_id": "risk_agent", "role": "risk", "decision": "REJECT",
        "confidence": 0.90, "evidence": base_facts, "rule_references": ["fraud-screen-v1"],
        "reasoning_summary": "Attempting whole-claim reject from risk screening.",
        "input_reference": binding,
    }
    spec_risk = assess_specialist(raw_risk_reject, "risk", base_facts, binding)
    assert spec_risk.validation_status == "INVALID"
    assert spec_risk.obligation_result == "UNCERTAIN"
    assert spec_risk.rule_consequence == "REVIEW"
    assert any("impossible_role_consequence:REJECT_not_permitted_for_risk" in r for r in spec_risk.reasons)

    # Evidence asserting REJECT
    ev_facts = InsuranceClaim().facts("evidence")
    ev_binding = {"artifact_id": "facts:evidence", "version": 1, "content_hash": "dummy"}
    raw_ev_reject = {
        "agent_id": "evidence_agent", "role": "evidence", "decision": "REJECT",
        "confidence": 0.85, "evidence": ev_facts, "rule_references": ["documents-v1"],
        "reasoning_summary": "Attempting reject for missing docs.",
        "input_reference": ev_binding,
    }
    spec_ev = assess_specialist(raw_ev_reject, "evidence", ev_facts, ev_binding)
    assert spec_ev.validation_status == "INVALID"
    assert spec_ev.obligation_result == "UNCERTAIN"
    assert spec_ev.rule_consequence == "REVIEW"

    # Policy asserting REVIEW
    pol_facts = InsuranceClaim().facts("policy")
    pol_binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "dummy"}
    raw_pol_review = {
        "agent_id": "policy_agent", "role": "policy", "decision": "REVIEW",
        "confidence": 0.90, "evidence": pol_facts, "rule_references": ["coverage-v1"],
        "reasoning_summary": "Attempting review for policy eligibility.",
        "input_reference": pol_binding,
    }
    spec_pol = assess_specialist(raw_pol_review, "policy", pol_facts, pol_binding)
    assert spec_pol.validation_status == "INVALID"
    assert spec_pol.obligation_result == "UNCERTAIN"
    assert spec_pol.rule_consequence == "REVIEW"


def test_missing_failed_stale_malformed_assessments():
    facts = InsuranceClaim().facts("policy")
    binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "hash1"}

    # Malformed output
    malformed = {"nonsense": True}
    spec_mal = assess_specialist(malformed, "policy", facts, binding)
    assert spec_mal.validation_status == "FAILED"
    assert spec_mal.obligation_result == "UNCERTAIN"
    assert spec_mal.rule_consequence == "REVIEW"

    # Execution error
    err_out = {"execution_error": "TimeoutError"}
    spec_err = assess_specialist(err_out, "policy", facts, binding)
    assert spec_err.validation_status == "FAILED"
    assert spec_err.obligation_result == "UNCERTAIN"
    assert spec_err.rule_consequence == "REVIEW"

    # Stale binding
    stale_out = {
        "agent_id": "policy_agent", "role": "policy", "decision": "APPROVE",
        "confidence": 0.90, "evidence": facts, "rule_references": ["coverage-v1"],
        "reasoning_summary": "Valid facts but stale binding.",
        "input_reference": {"artifact_id": "facts:policy", "version": 0, "content_hash": "old_hash"},
    }
    spec_stale = assess_specialist(stale_out, "policy", facts, binding)
    assert spec_stale.validation_status == "INVALID"
    assert spec_stale.obligation_result == "UNCERTAIN"
    assert "stale_or_unbound_input" in spec_stale.reasons

    # Output decision conflicting with deterministic rule
    conflict_out = {
        "agent_id": "policy_agent", "role": "policy", "decision": "REJECT",
        "confidence": 0.90, "evidence": facts, "rule_references": ["coverage-v1"],
        "reasoning_summary": "Asserting reject on clean eligible claim.",
        "input_reference": binding,
    }
    spec_conf = assess_specialist(conflict_out, "policy", facts, binding)
    assert spec_conf.validation_status == "INVALID"
    assert spec_conf.obligation_result == "UNCERTAIN"
    assert "decision_conflicts_with_rule" in spec_conf.reasons


def test_advisory_confidence_containment():
    # Advisory consensus calculates weighted diagnostic decision, but CANNOT override truth table
    claim = replace(InsuranceClaim(), hospitalization_covered=False)
    workflow = SpecialistCompositionAdjudication(claim)
    result = workflow.result()

    # Truth table composed decision is REJECT
    assert result["decision"] == "REJECT"

    # Advisory consensus candidate was APPROVE (due to 3 non-policy specialists voting APPROVE)
    adv = result["advisory_consensus"]
    assert adv["diagnostic_candidate"] == "APPROVE"
    assert adv["role_authority"] == "ADVISORY_DIAGNOSTIC_ONLY"
    assert adv["overrode_decision"] is False
    assert adv["agreed_with_composed"] is False


def test_selective_recovery_reruns_only_affected_roles():
    claim = InsuranceClaim()
    workflow = SpecialistCompositionAdjudication(claim)
    assert workflow.result()["decision"] == "APPROVE"
    assert all(count == 1 for count in workflow.executions.values())

    # Invalidate evidence_agent
    outcome = workflow.recover_agent("evidence_agent")
    assert workflow.executions["evidence"] == 2
    assert workflow.executions["policy"] == 1
    assert workflow.executions["risk"] == 1
    assert workflow.executions["compliance"] == 1

    # Invalidate via claim fact change (only risk fact changes)
    changed_claim = replace(claim, duplicate_claim=True)
    rec = workflow.update_claim(changed_claim)
    assert rec is not None
    assert workflow.executions["risk"] == 2
    assert workflow.executions["policy"] == 1
    assert workflow.executions["evidence"] == 2
    assert workflow.executions["compliance"] == 1

    # Decision updated correctly
    assert workflow.result()["decision"] == "REVIEW"


def test_legacy_mode_preservation():
    # Legacy mode runs legacy weighted consensus + mandatory gate
    claim = replace(InsuranceClaim(), hospitalization_covered=False)
    legacy_wf = InsuranceAdjudication(claim, mode="legacy")
    result = legacy_wf.result()

    # In legacy mode, 3-to-1 voting override caused the gate to defer to REVIEW (the 42 false reviews bug)
    assert result["decision"] == "REVIEW"
    assert result["gate"] == "REVIEW_REQUIRED"
    assert "consensus" in result

    # Check certificate rule version
    cert_ref = legacy_wf.system.latest("round-certificate").ref
    cert = legacy_wf.system.certificate(cert_ref)
    assert cert.method == "insurance-round-v1"


def test_export_payload_consistency():
    claim = InsuranceClaim()
    workflow = SpecialistCompositionAdjudication(claim)
    exported = workflow.export()

    assert exported["synthetic"] is True
    assert exported["final"]["mode"] == "specialist-composition"
    assert len(exported["agents"]) == 4
    assert len(exported["events"]) > 0
    assert exported["round"]["contract_version"] == COMPOSITION_CONTRACT_VERSION
    assert workflow.ledger.verify(exported["events"], exported["dag_checkpoint"])
